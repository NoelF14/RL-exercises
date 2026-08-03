"""Aggregate and plot Phase 0 CSV artifacts without importing CARL."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SPLIT_ORDER = ("train", "id_test", "ood_low", "ood_high")
METHOD_ORDER = ("hidden", "oracle")
EPISODE_COLUMNS = {
    "method",
    "seed",
    "context_feature",
    "context_value",
    "split",
    "episode_index",
    "episode_seed",
    "return",
    "episode_length",
}
TRAINING_COLUMNS = {
    "method",
    "seed",
    "context_feature",
    "environment_steps",
    "episode_index",
    "episode_return",
    "episode_length",
}


def analyze_results(results_root: Path, output_dir: Path) -> dict[str, Path]:
    """Create Phase 0 aggregate tables and plots from saved CSV files only."""
    episodes = _load_atomic_csvs(results_root, "episode_returns.csv", EPISODE_COLUMNS)
    training = _load_atomic_csvs(results_root, "training_metrics.csv", TRAINING_COLUMNS)
    output_dir.mkdir(parents=True, exist_ok=True)

    screening = (
        episodes.groupby(["context_feature", "method", "seed", "split"], sort=True)
        .agg(
            number_contexts=("context_value", "nunique"),
            number_episodes=("return", "size"),
            mean_return=("return", "mean"),
            std_return=("return", "std"),
            median_return=("return", "median"),
        )
        .reset_index()
    )
    screening["std_return"] = screening["std_return"].fillna(0.0)

    paired_gaps = _paired_gaps(screening)
    context_pairs = _context_pairs(episodes)

    paths = {
        "screening": output_dir / "phase0_screening_runs.csv",
        "paired_gaps": output_dir / "phase0_paired_gaps.csv",
        "context_pairs": output_dir / "phase0_context_pairs.csv",
        "mean_returns_plot": output_dir / "phase0_mean_returns.png",
        "gaps_plot": output_dir / "phase0_paired_gaps.png",
        "context_plot": output_dir / "phase0_return_vs_context.png",
        "training_plot": output_dir / "phase0_training_curves.png",
    }
    screening.to_csv(paths["screening"], index=False)
    paired_gaps.to_csv(paths["paired_gaps"], index=False)
    context_pairs.to_csv(paths["context_pairs"], index=False)
    _plot_mean_returns(screening, paths["mean_returns_plot"])
    _plot_gaps(paired_gaps, paths["gaps_plot"])
    _plot_context_returns(context_pairs, paths["context_plot"])
    _plot_training(training, paths["training_plot"])
    return paths


def _load_atomic_csvs(
    results_root: Path, filename: str, required_columns: set[str]
) -> pd.DataFrame:
    files = sorted(results_root.glob(f"*/*/seed_*/{filename}"))
    if not files:
        raise FileNotFoundError(f"No atomic {filename} files found below {results_root}")
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        missing = required_columns - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"{path} contains no data rows")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _paired_gaps(screening: pd.DataFrame) -> pd.DataFrame:
    index = ["context_feature", "seed", "split"]
    pivot = screening.pivot(index=index, columns="method", values="mean_return")
    missing = set(METHOD_ORDER) - set(pivot.columns)
    if missing:
        raise ValueError(f"Cannot pair methods; missing {sorted(missing)}")
    paired = pivot.reset_index().rename(
        columns={"hidden": "mean_return_hidden", "oracle": "mean_return_oracle"}
    )
    paired["oracle_gap"] = (
        paired["mean_return_oracle"] - paired["mean_return_hidden"]
    )
    return paired.sort_values(index).reset_index(drop=True)


def _context_pairs(episodes: pd.DataFrame) -> pd.DataFrame:
    keys = ["context_feature", "seed", "split", "context_value"]
    context = (
        episodes.groupby([*keys, "method"], sort=True)
        .agg(
            number_episodes=("return", "size"),
            mean_return=("return", "mean"),
            std_return=("return", "std"),
            median_return=("return", "median"),
        )
        .reset_index()
    )
    context["std_return"] = context["std_return"].fillna(0.0)
    hidden = context[context["method"] == "hidden"].drop(columns="method").rename(
        columns={column: f"{column}_hidden" for column in context.columns if column not in keys + ["method"]}
    )
    oracle = context[context["method"] == "oracle"].drop(columns="method").rename(
        columns={column: f"{column}_oracle" for column in context.columns if column not in keys + ["method"]}
    )
    paired = hidden.merge(oracle, on=keys, how="inner", validate="one_to_one")
    paired["oracle_gap"] = (
        paired["mean_return_oracle"] - paired["mean_return_hidden"]
    )
    return paired.sort_values(keys).reset_index(drop=True)


def _plot_mean_returns(screening: pd.DataFrame, path: Path) -> None:
    features = sorted(screening["context_feature"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for ax, split in zip(axes.flat, SPLIT_ORDER, strict=True):
        subset = screening[screening["split"] == split]
        for method_index, method in enumerate(METHOD_ORDER):
            method_rows = subset[subset["method"] == method]
            for seed_index, seed in enumerate(sorted(method_rows["seed"].unique())):
                rows = method_rows[method_rows["seed"] == seed].set_index("context_feature")
                x = np.arange(len(features)) + (method_index - 0.5) * 0.18
                ax.scatter(x, [rows.loc[f, "mean_return"] for f in features], label=f"{method}, seed {seed}")
        ax.set_title(split)
        ax.set_xticks(np.arange(len(features)), features)
        ax.set_ylabel("Mean episode return")
        ax.grid(alpha=0.25)
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_gaps(gaps: pd.DataFrame, path: Path) -> None:
    features = sorted(gaps["context_feature"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for ax, split in zip(axes.flat, SPLIT_ORDER, strict=True):
        subset = gaps[gaps["split"] == split]
        for seed_index, seed in enumerate(sorted(subset["seed"].unique())):
            rows = subset[subset["seed"] == seed].set_index("context_feature")
            x = np.arange(len(features)) + (seed_index - 0.5) * 0.14
            ax.scatter(x, [rows.loc[f, "oracle_gap"] for f in features], label=f"seed {seed}")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(split)
        ax.set_xticks(np.arange(len(features)), features)
        ax.set_ylabel("Oracle minus hidden return")
        ax.grid(alpha=0.25)
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_context_returns(context_pairs: pd.DataFrame, path: Path) -> None:
    features = sorted(context_pairs["context_feature"].unique())
    fig, axes = plt.subplots(len(features), len(SPLIT_ORDER), figsize=(16, 4 * len(features)), squeeze=False)
    for row_index, feature in enumerate(features):
        for column_index, split in enumerate(SPLIT_ORDER):
            ax = axes[row_index, column_index]
            subset = context_pairs[
                (context_pairs["context_feature"] == feature)
                & (context_pairs["split"] == split)
            ]
            for method in METHOD_ORDER:
                for seed in sorted(subset["seed"].unique()):
                    rows = subset[subset["seed"] == seed].sort_values("context_value")
                    ax.plot(rows["context_value"], rows[f"mean_return_{method}"], marker="o", label=f"{method}, seed {seed}")
            ax.set_title(f"{feature}: {split}")
            ax.set_xlabel("Context value")
            ax.set_ylabel("Mean episode return")
            ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_training(training: pd.DataFrame, path: Path) -> None:
    features = sorted(training["context_feature"].unique())
    fig, axes = plt.subplots(len(features), 1, figsize=(11, 4 * len(features)), squeeze=False)
    for ax, feature in zip(axes.flat, features, strict=True):
        subset = training[training["context_feature"] == feature]
        for method in METHOD_ORDER:
            for seed in sorted(subset["seed"].unique()):
                rows = subset[(subset["method"] == method) & (subset["seed"] == seed)].sort_values("environment_steps")
                smoothed = rows["episode_return"].rolling(10, min_periods=1).mean()
                ax.plot(rows["environment_steps"], smoothed, label=f"{method}, seed {seed}")
        ax.set_title(feature)
        ax.set_xlabel("Environment steps")
        ax.set_ylabel("Episode return (10-episode mean)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.results_root / "analysis"
    paths = analyze_results(args.results_root, output_dir)
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    main()
