"""Analyze saved Phase 0 diagnostic CSV/config artifacts without importing CARL."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


EXPECTED_RUNS = {
    "default_100k": (100_000, "default", "default"),
    "default_300k": (300_000, "default", "default"),
    "specialist_low_300k": (300_000, "specialist", "low"),
    "specialist_center_300k": (300_000, "specialist", "center"),
    "specialist_high_300k": (300_000, "specialist", "high"),
    "contextual_300k": (300_000, "contextual", "contextual"),
}
EPISODE_COLUMNS = {
    "run_id", "method", "seed", "context_feature", "context_value", "split",
    "context_id", "episode_index", "episode_seed", "return", "episode_length",
}


def analyze_diagnostic(results_root: Path, output_dir: Path) -> dict[str, Path]:
    """Write seed-replicate summaries, descriptive tables, and diagnostic plots."""
    seed_results = _load_seed_results(results_root)
    _validate_complete_matrix(seed_results)
    output_dir.mkdir(parents=True, exist_ok=True)

    defaults = seed_results[seed_results["diagnostic"] == "default"].copy()
    default_paired = _pair_default_budgets(defaults)
    default_summary = _seed_summary(defaults, ["total_timesteps", "split"])

    specialists = seed_results[seed_results["diagnostic"] == "specialist"].copy()
    specialist_summary = _seed_summary(specialists, ["setting", "context_value", "split"])

    contextual = seed_results[seed_results["diagnostic"] == "contextual"].copy()
    contextual_id = contextual[contextual["split"].isin(["train", "id_test"])].copy()
    contextual_comparison = _pair_contextual_methods(contextual_id)
    contextual_summary = _seed_summary(contextual_id, ["method", "split"])

    ood_seed = contextual[contextual["split"].isin(["ood_low", "ood_high"])].copy()
    ood_seed["analysis_role"] = "descriptive_only"
    ood_seed["eligible_for_tuning_or_selection"] = False
    ood_summary = _seed_summary(ood_seed, ["method", "split"])
    ood_summary["analysis_role"] = "descriptive_only"
    ood_summary["eligible_for_tuning_or_selection"] = False

    paths = {
        "seed_results": output_dir / "diagnostic_seed_results.csv",
        "default_paired": output_dir / "default_100k_vs_300k_by_seed.csv",
        "default_summary": output_dir / "default_context_summary.csv",
        "specialist_summary": output_dir / "specialist_summary.csv",
        "contextual_comparison": output_dir / "contextual_hidden_vs_oracle_by_seed.csv",
        "contextual_summary": output_dir / "contextual_train_id_summary.csv",
        "ood_seed": output_dir / "contextual_ood_descriptive_by_seed.csv",
        "ood_summary": output_dir / "contextual_ood_descriptive_summary.csv",
        "default_plot": output_dir / "default_100k_vs_300k.png",
        "specialist_plot": output_dir / "specialist_performance.png",
        "contextual_plot": output_dir / "contextual_train_id.png",
        "ood_plot": output_dir / "contextual_ood_descriptive.png",
    }
    for key, frame in (
        ("seed_results", seed_results),
        ("default_paired", default_paired),
        ("default_summary", default_summary),
        ("specialist_summary", specialist_summary),
        ("contextual_comparison", contextual_comparison),
        ("contextual_summary", contextual_summary),
        ("ood_seed", ood_seed),
        ("ood_summary", ood_summary),
    ):
        frame.to_csv(paths[key], index=False)
    _plot_defaults(defaults, paths["default_plot"])
    _plot_specialists(specialists, paths["specialist_plot"])
    _plot_contextual(contextual_id, paths["contextual_plot"])
    _plot_ood(ood_seed, paths["ood_plot"])
    return paths


def _load_seed_results(results_root: Path) -> pd.DataFrame:
    files = sorted(results_root.glob("*/*/*/seed_*/episode_returns.csv"))
    if not files:
        raise FileNotFoundError(f"No diagnostic episode_returns.csv files below {results_root}")
    frames = []
    for path in files:
        config_path = path.parent / "resolved_config.yaml"
        if not config_path.is_file():
            raise ValueError(f"Missing saved config artifact: {config_path}")
        with config_path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        if not isinstance(config, dict):
            raise ValueError(f"Invalid saved config artifact: {config_path}")
        experiment = str(config["experiment"]["name"])
        if experiment not in EXPECTED_RUNS:
            raise ValueError(f"Unexpected diagnostic experiment {experiment!r} in {path}")
        total_timesteps, diagnostic, setting = EXPECTED_RUNS[experiment]
        if int(config["training"]["total_timesteps"]) != total_timesteps:
            raise ValueError(f"Saved timestep budget does not match {experiment}: {config_path}")
        run = config.get("run", {})
        frame = pd.read_csv(path)
        missing = EPISODE_COLUMNS - set(frame.columns)
        if missing or frame.empty:
            raise ValueError(f"{path} is empty or missing columns: {sorted(missing)}")
        expected = {
            "method": str(run.get("observation_mode")),
            "seed": int(run.get("seed")),
            "context_feature": str(run.get("context_feature")),
        }
        for column, value in expected.items():
            if set(frame[column]) != {value}:
                raise ValueError(f"{path} does not match resolved_config.yaml field {column}")
        if expected["context_feature"] != "length":
            raise ValueError(f"Diagnostic analyzer accepts only length runs: {path}")
        frame["experiment"] = experiment
        frame["diagnostic"] = diagnostic
        frame["setting"] = setting
        frame["total_timesteps"] = total_timesteps
        frames.append(frame)

    episodes = pd.concat(frames, ignore_index=True)
    keys = [
        "experiment", "diagnostic", "setting", "total_timesteps", "method", "seed",
        "context_feature", "split",
    ]
    results = (
        episodes.groupby(keys, sort=True)
        .agg(
            number_contexts=("context_value", "nunique"),
            number_episodes=("return", "size"),
            context_value_min=("context_value", "min"),
            context_value_max=("context_value", "max"),
            mean_return=("return", "mean"),
        )
        .reset_index()
    )
    results["context_value"] = np.where(
        results["context_value_min"] == results["context_value_max"],
        results["context_value_min"],
        np.nan,
    )
    return results.sort_values(keys).reset_index(drop=True)


def _validate_complete_matrix(seed_results: pd.DataFrame) -> None:
    runs = seed_results[["experiment", "method", "seed"]].drop_duplicates()
    expected = set()
    for experiment in EXPECTED_RUNS:
        methods = ("hidden", "oracle") if experiment == "contextual_300k" else ("hidden",)
        expected.update((experiment, method, seed) for method in methods for seed in (0, 1))
    actual = set(runs.itertuples(index=False, name=None))
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"Incomplete or unexpected diagnostic matrix; missing={missing}, extra={extra}")

    fixed = seed_results[seed_results["diagnostic"].isin(["default", "specialist"])]
    if set(fixed["split"]) != {"train"} or not (fixed["number_contexts"] == 1).all():
        raise ValueError("Default and specialist runs must evaluate one fixed train context only")
    contextual = seed_results[seed_results["diagnostic"] == "contextual"]
    if set(contextual["split"]) != {"train", "id_test", "ood_low", "ood_high"}:
        raise ValueError("Contextual runs must preserve train, ID, OOD-low, and OOD-high")
    expected_counts = {"train": 9, "id_test": 8, "ood_low": 5, "ood_high": 5}
    for split, count in expected_counts.items():
        if not (contextual[contextual["split"] == split]["number_contexts"] == count).all():
            raise ValueError(f"Contextual {split} results must contain exactly {count} contexts")


def _seed_summary(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    summary = (
        frame.groupby(groups, sort=True)
        .agg(
            n_seeds=("seed", "nunique"),
            mean_of_seed_means=("mean_return", "mean"),
            std_of_seed_means=("mean_return", "std"),
            min_seed_mean=("mean_return", "min"),
            max_seed_mean=("mean_return", "max"),
        )
        .reset_index()
    )
    if not (summary["n_seeds"] == 2).all():
        raise ValueError("Every diagnostic summary requires exactly two seed replicates")
    return summary


def _pair_default_budgets(defaults: pd.DataFrame) -> pd.DataFrame:
    pivot = defaults.pivot(index=["seed", "split"], columns="total_timesteps", values="mean_return")
    if set(pivot.columns) != {100_000, 300_000}:
        raise ValueError("Cannot compare default-context 100k and 300k runs")
    paired = pivot.reset_index().rename(columns={100_000: "mean_return_100k", 300_000: "mean_return_300k"})
    paired["delta_300k_minus_100k"] = paired["mean_return_300k"] - paired["mean_return_100k"]
    return paired


def _pair_contextual_methods(contextual: pd.DataFrame) -> pd.DataFrame:
    pivot = contextual.pivot(index=["seed", "split"], columns="method", values="mean_return")
    if set(pivot.columns) != {"hidden", "oracle"}:
        raise ValueError("Cannot compare contextual hidden and oracle runs")
    paired = pivot.reset_index().rename(columns={"hidden": "mean_return_hidden", "oracle": "mean_return_oracle"})
    paired["oracle_minus_hidden"] = paired["mean_return_oracle"] - paired["mean_return_hidden"]
    return paired.sort_values(["split", "seed"]).reset_index(drop=True)


def _plot_defaults(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for seed, rows in frame.groupby("seed"):
        rows = rows.sort_values("total_timesteps")
        ax.plot(rows["total_timesteps"] / 1000, rows["mean_return"], marker="o", label=f"seed {seed}")
    ax.set(xlabel="Training steps (thousands)", ylabel="Mean evaluation return", title="Fixed default length")
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def _plot_specialists(frame: pd.DataFrame, path: Path) -> None:
    order = ["low", "center", "high"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for seed, rows in frame.groupby("seed"):
        rows = rows.set_index("setting").loc[order]
        ax.plot(order, rows["mean_return"], marker="o", label=f"seed {seed}")
    ax.set(xlabel="Specialist training/evaluation length", ylabel="Mean evaluation return", title="Single-context specialists")
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def _plot_contextual(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, split in zip(axes, ("train", "id_test"), strict=True):
        subset = frame[frame["split"] == split]
        for seed, rows in subset.groupby("seed"):
            rows = rows.set_index("method").loc[["hidden", "oracle"]]
            ax.plot(["hidden", "oracle"], rows["mean_return"], marker="o", label=f"seed {seed}")
        ax.set(title=split, ylabel="Mean evaluation return"); ax.grid(alpha=0.25)
    axes[0].legend(); fig.suptitle("300k contextual comparison"); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def _plot_ood(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, split in zip(axes, ("ood_low", "ood_high"), strict=True):
        subset = frame[frame["split"] == split]
        for seed, rows in subset.groupby("seed"):
            rows = rows.set_index("method").loc[["hidden", "oracle"]]
            ax.plot(["hidden", "oracle"], rows["mean_return"], marker="o", label=f"seed {seed}")
        ax.set(title=f"{split} (descriptive only)", ylabel="Mean evaluation return"); ax.grid(alpha=0.25)
    axes[0].legend(); fig.suptitle("OOD reporting—not for tuning or selection"); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results/phase0_diagnostic"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.results_root / "analysis"
    for path in analyze_diagnostic(args.results_root, output_dir).values():
        print(path)


if __name__ == "__main__":
    main()
