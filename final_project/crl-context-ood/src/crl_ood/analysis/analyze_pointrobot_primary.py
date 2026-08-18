"""Result-only analysis for the frozen PointRobot primary experiment.

Only persisted CSV, JSON, YAML, and NumPy inputs are read. Five end-to-end
policy seeds are the sole independent experimental replicates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import yaml

METHODS = ("no_context", "oracle", "vae", "contrastive")
LEARNED = ("vae", "contrastive")
SEEDS = (0, 1, 2, 3, 4)
SPLITS = {"train": (-0.6, -0.3, 0.0, 0.3, 0.6), "id": (-0.45, -0.15, 0.15, 0.45),
          "ood_left": (-1.0, -0.8), "ood_right": (0.8, 1.0)}
METRICS = ("mean_return", "return_std", "success_rate", "mean_final_distance",
           "mean_minimum_distance", "mean_first_success_timestep", "evaluation_episode_count")
OUTPUTS = ("primary_context_results.csv", "primary_summary_by_seed.csv", "primary_summary_across_seeds.csv",
    "primary_paired_return_gaps.csv", "primary_paired_success_gaps.csv", "primary_oracle_gap_closure.csv",
    "primary_ood_degradation.csv", "primary_near_far_ood_summary.csv", "primary_encoder_checkpoint_manifest.csv",
    "primary_training_budget_verification.csv", "primary_findings.json", "primary_findings.md",
    "return_by_goal.png", "success_by_goal.png", "final_distance_by_goal.png", "oracle_gap_closure.png",
    "id_vs_ood_return.png", "near_vs_far_ood.png", "paired_vae_contrastive_difference.png", "training_curves.png")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() != ".csv":
        raise ValueError("analysis input must be a persisted CSV")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: Iterable[str] | None = None) -> None:
    fields = list(columns or (rows[0].keys() if rows else ()))
    if not fields:
        raise ValueError(f"no schema available for {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def paired_bootstrap_interval(differences: Iterable[float], analysis_seed: int,
                              resamples: int = 10000) -> tuple[float, float]:
    values = np.asarray(list(differences), dtype=float)
    if values.shape != (5,):
        raise ValueError("paired bootstrap requires exactly five seed-level differences")
    if resamples != 10000:
        raise ValueError("primary bootstrap is frozen at 10000 resamples")
    rng = np.random.default_rng(int(analysis_seed))
    sampled = values[rng.integers(0, 5, size=(resamples, 5))].mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return float(low), float(high)


def _float(row: dict[str, Any], key: str) -> float:
    value = row[key]
    return math.nan if value in (None, "", "nan", "NaN") else float(value)


def _mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(np.nanmean(array)) if not np.isnan(array).all() else math.nan


def _std(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[~np.isnan(array)]
    return float(np.std(array, ddof=1)) if len(array) > 1 else math.nan


def _collect(root: Path, spec: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, provenances = [], []
    expected_checksum = None
    for method in METHODS:
        for seed in SEEDS:
            run = root / "downstream" / method / f"seed_{seed}"
            context_path, provenance_path = run / "context_metrics.csv", run / "provenance.json"
            if not context_path.is_file() or not provenance_path.is_file():
                raise RuntimeError(f"primary analysis unavailable: incomplete downstream job {method} seed {seed}")
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            required = {"method", "encoder_seed", "policy_seed", "requested_timesteps",
                "actual_complete_rollout_timesteps", "rollout_quantum", "dataset_checksum",
                "encoder_checkpoint_path", "encoder_checkpoint_sha256", "source_commit",
                "configuration_checksum", "selection_split", "ood_role"}
            if required - set(provenance):
                raise ValueError(f"incomplete provenance for {method} seed {seed}: {sorted(required - set(provenance))}")
            if provenance["method"] != method or int(provenance["policy_seed"]) != seed:
                raise ValueError("downstream method/policy seed provenance mismatch")
            if method in LEARNED:
                if int(provenance["encoder_seed"]) != seed or not provenance["encoder_checkpoint_path"] or not provenance["encoder_checkpoint_sha256"]:
                    raise ValueError("learned seed mapping/checkpoint provenance mismatch")
            if provenance["selection_split"] != "training contexts only" or provenance["ood_role"] != "descriptive/scientific evaluation only":
                raise ValueError("selection or OOD provenance violates the frozen specification")
            if provenance["source_commit"] != spec["experiment"]["source_commit"]:
                raise ValueError("source commit provenance mismatch")
            expected_checksum = expected_checksum or provenance["dataset_checksum"]
            if provenance["dataset_checksum"] != expected_checksum:
                raise ValueError("dataset checksum differs across downstream jobs")
            requested, actual, quantum = (int(provenance[key]) for key in
                ("requested_timesteps", "actual_complete_rollout_timesteps", "rollout_quantum"))
            expected_actual = ((requested + quantum - 1) // quantum) * quantum
            if requested != 200000 or actual != expected_actual:
                raise ValueError("requested versus complete-rollout timestep validation failed")
            job_rows = _read_csv(context_path)
            expected_pairs = {(split, angle) for split, angles in SPLITS.items() for angle in angles}
            actual_pairs = {(row["split"], float(row["goal_angle"])) for row in job_rows}
            if actual_pairs != expected_pairs or len(job_rows) != len(expected_pairs):
                raise ValueError("context metrics do not preserve every split and individual goal angle")
            for row in job_rows:
                if int(row["evaluation_episode_count"]) != int(spec["evaluation"]["episodes_per_context"]):
                    raise ValueError("context evaluation episode count mismatch")
                rows.append({"method": method, "policy_seed": seed,
                    "encoder_seed": int(row["encoder_seed"]) if row["encoder_seed"] not in ("", "None") else "",
                    "split": row["split"], "goal_angle": float(row["goal_angle"]),
                    **{metric: _float(row, metric) for metric in METRICS}})
            provenances.append(provenance)
    if len({int(item["policy_seed"]) for item in provenances}) != 5:
        raise ValueError("exactly five unique end-to-end seeds are required")
    for method in LEARNED:
        paths = {item["encoder_checkpoint_path"] for item in provenances if item["method"] == method}
        if len(paths) != 5:
            raise ValueError("one learned checkpoint may not be reused for all policy seeds")
    return rows, provenances


def _summaries(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_seed = []
    for method in METHODS:
        for seed in SEEDS:
            for split in SPLITS:
                selected = [row for row in rows if row["method"] == method and row["policy_seed"] == seed and row["split"] == split]
                by_seed.append({"method": method, "policy_seed": seed, "split": split,
                    **{metric: _mean(_float(row, metric) for row in selected) for metric in METRICS}})
    across = []
    for method in METHODS:
        for split in SPLITS:
            selected = [row for row in by_seed if row["method"] == method and row["split"] == split]
            record: dict[str, Any] = {"method": method, "split": split, "independent_seed_count": 5}
            for metric in METRICS:
                values = [_float(row, metric) for row in selected]
                record[f"{metric}_seed_values"] = json.dumps(values, separators=(",", ":"))
                record[f"{metric}_mean_across_seeds"] = _mean(values)
                record[f"{metric}_std_across_seeds"] = _std(values)
            across.append(record)
    return by_seed, across


def _lookup(by_seed: list[dict[str, Any]], method: str, seed: int, split: str, metric: str) -> float:
    return _float(next(row for row in by_seed if row["method"] == method and row["policy_seed"] == seed and row["split"] == split), metric)


def _paired(by_seed: list[dict[str, Any]], metric: str, spec: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    comparisons = (("oracle", "no_context"), ("vae", "no_context"),
                   ("contrastive", "no_context"), ("vae", "contrastive"))
    for split in SPLITS:
        for first, second in comparisons:
            values = [_lookup(by_seed, first, seed, split, metric) - _lookup(by_seed, second, seed, split, metric) for seed in SEEDS]
            low, high = paired_bootstrap_interval(values, int(spec["analysis"]["analysis_seed"]),
                                                   int(spec["analysis"]["bootstrap_resamples"]))
            rows.append({"split": split, "comparison": f"{first}_minus_{second}", "seed_count": 5,
                "seed_0": values[0], "seed_1": values[1], "seed_2": values[2], "seed_3": values[3], "seed_4": values[4],
                "mean_difference": _mean(values), "std_difference": _std(values), "bootstrap_95_low": low,
                "bootstrap_95_high": high, "interval_label": "low-sample descriptive interval",
                "replicate_unit": "paired_end_to_end_seed"})
    return rows


def _derived(rows: list[dict[str, Any]], by_seed: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    closure, degradation, near_far = [], [], []
    for method in LEARNED:
        for seed in SEEDS:
            for split in ("train", "id"):
                baseline = _lookup(by_seed, "no_context", seed, split, "mean_return")
                oracle = _lookup(by_seed, "oracle", seed, split, "mean_return")
                learned = _lookup(by_seed, method, seed, split, "mean_return")
                closure.append({"method": method, "policy_seed": seed, "split": split,
                    "oracle_gap_closure": (learned - baseline) / (oracle - baseline) if oracle != baseline else math.nan})
    for method in METHODS:
        for seed in SEEDS:
            id_return = _lookup(by_seed, method, seed, "id", "mean_return")
            for side in ("ood_left", "ood_right"):
                side_return = _lookup(by_seed, method, seed, side, "mean_return")
                degradation.append({"method": method, "policy_seed": seed, "ood_side": side,
                    "id_mean_return": id_return, "ood_mean_return": side_return,
                    "ood_return_change_from_id": side_return - id_return})
            for side in ("ood_left", "ood_right"):
                for distance, angle_abs in (("near", 0.8), ("far", 1.0)):
                    selected = [row for row in rows if row["method"] == method and row["policy_seed"] == seed
                                and row["split"] == side and np.isclose(abs(row["goal_angle"]), angle_abs)]
                    if len(selected) != 1:
                        raise ValueError("near/far OOD grouping must retain exactly one angle per side")
                    row = selected[0]
                    near_far.append({"method": method, "policy_seed": seed, "ood_side": side,
                        "distance_group": distance, "goal_angle": row["goal_angle"],
                        "mean_return": row["mean_return"], "success_rate": row["success_rate"],
                        "mean_final_distance": row["mean_final_distance"]})
    for collection, group_keys, value_key in (
        (closure, ("method", "split"), "oracle_gap_closure"),
        (degradation, ("method", "ood_side"), "ood_return_change_from_id"),
        (near_far, ("method", "ood_side", "distance_group"), "mean_return"),
    ):
        for item in collection:
            selected = [candidate[value_key] for candidate in collection
                        if all(candidate[key] == item[key] for key in group_keys)]
            item["seed_values"] = json.dumps(selected, separators=(",", ":"))
            item["mean_across_seeds"] = _mean(selected)
            item["std_across_seeds"] = _std(selected)
    return closure, degradation, near_far


def _plots(root: Path, context: list[dict[str, Any]], by_seed: list[dict[str, Any]],
           closure: list[dict[str, Any]], near_far: list[dict[str, Any]]) -> None:
    def goal_plot(metric: str, filename: str, ylabel: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for method in METHODS:
            angles = sorted({row["goal_angle"] for row in context})
            means = [_mean(row[metric] for row in context if row["method"] == method and row["goal_angle"] == angle) for angle in angles]
            ax.plot(angles, means, marker="o", label=method)
        ax.set(xlabel="goal angle", ylabel=ylabel); ax.legend(); fig.tight_layout(); fig.savefig(root / filename, dpi=150); plt.close(fig)
    goal_plot("mean_return", "return_by_goal.png", "mean return")
    goal_plot("success_rate", "success_by_goal.png", "success rate")
    goal_plot("mean_final_distance", "final_distance_by_goal.png", "mean final distance")
    fig, ax = plt.subplots(figsize=(6, 4))
    for method in LEARNED:
        ax.bar([f"{method}\ntrain", f"{method}\nID"], [_mean(r["oracle_gap_closure"] for r in closure if r["method"] == method and r["split"] == split) for split in ("train", "id")])
    ax.set(ylabel="oracle gap closure"); fig.tight_layout(); fig.savefig(root / "oracle_gap_closure.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    labels, values = [], []
    for method in METHODS:
        for split in ("id", "ood_left", "ood_right"):
            labels.append(f"{method}\n{split}"); values.append(_mean(_lookup(by_seed, method, seed, split, "mean_return") for seed in SEEDS))
    ax.bar(labels, values); ax.tick_params(axis="x", rotation=45); ax.set(ylabel="return"); fig.tight_layout(); fig.savefig(root / "id_vs_ood_return.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    labels, values = [], []
    for method in METHODS:
        for group in ("near", "far"):
            labels.append(f"{method}\n{group}"); values.append(_mean(r["mean_return"] for r in near_far if r["method"] == method and r["distance_group"] == group))
    ax.bar(labels, values); ax.tick_params(axis="x", rotation=45); ax.set(ylabel="OOD return"); fig.tight_layout(); fig.savefig(root / "near_vs_far_ood.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for split in SPLITS:
        values = [_lookup(by_seed, "vae", seed, split, "mean_return") - _lookup(by_seed, "contrastive", seed, split, "mean_return") for seed in SEEDS]
        ax.plot(SEEDS, values, marker="o", label=split)
    ax.axhline(0, color="black", linewidth=.8); ax.set(xlabel="paired seed", ylabel="VAE - contrastive return"); ax.legend(); fig.tight_layout(); fig.savefig(root / "paired_vae_contrastive_difference.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    found = False
    for method in METHODS:
        for seed in SEEDS:
            path = root / "downstream" / method / f"seed_{seed}" / "training_progress.csv"
            if path.is_file():
                progress = _read_csv(path); found = found or bool(progress)
                ax.plot([float(r["timesteps"]) for r in progress], [float(r["mean_recent_episode_return"]) for r in progress], alpha=.35, label=f"{method}-{seed}")
    ax.set(xlabel="complete-rollout timesteps", ylabel="recent training return")
    if found: ax.legend(ncol=4, fontsize=6)
    fig.tight_layout(); fig.savefig(root / "training_curves.png", dpi=150); plt.close(fig)


def analyze(results_dir: str | Path, spec_path: str | Path) -> Path:
    root, config_path = Path(results_dir), Path(spec_path)
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    configuration_checksum = hashlib.sha256(config_path.read_bytes()).hexdigest()
    rows, provenances = _collect(root, spec)
    if any(item["configuration_checksum"] != configuration_checksum for item in provenances):
        raise ValueError("downstream configuration checksum does not match the frozen specification")
    by_seed, across = _summaries(rows)
    return_gaps, success_gaps = _paired(by_seed, "mean_return", spec), _paired(by_seed, "success_rate", spec)
    closure, degradation, near_far = _derived(rows, by_seed)
    checkpoint_rows = [{key: item[key] for key in ("method", "encoder_seed", "policy_seed", "dataset_checksum",
        "encoder_checkpoint_path", "encoder_checkpoint_sha256", "source_commit", "configuration_checksum")}
        for item in provenances if item["method"] in LEARNED]
    budget_rows = [{"method": item["method"], "policy_seed": item["policy_seed"],
        "requested_timesteps": item["requested_timesteps"], "actual_complete_rollout_timesteps": item["actual_complete_rollout_timesteps"],
        "rollout_quantum": item["rollout_quantum"], "valid_complete_rollout_budget": True} for item in provenances]
    _write_csv(root / "primary_context_results.csv", rows)
    _write_csv(root / "primary_summary_by_seed.csv", by_seed)
    _write_csv(root / "primary_summary_across_seeds.csv", across)
    _write_csv(root / "primary_paired_return_gaps.csv", return_gaps)
    _write_csv(root / "primary_paired_success_gaps.csv", success_gaps)
    _write_csv(root / "primary_oracle_gap_closure.csv", closure)
    _write_csv(root / "primary_ood_degradation.csv", degradation)
    _write_csv(root / "primary_near_far_ood_summary.csv", near_far)
    _write_csv(root / "primary_encoder_checkpoint_manifest.csv", checkpoint_rows)
    _write_csv(root / "primary_training_budget_verification.csv", budget_rows)
    oracle_control = all(_mean(_lookup(by_seed, "oracle", seed, split, "mean_return") for seed in SEEDS) >
                         _mean(_lookup(by_seed, "no_context", seed, split, "mean_return") for seed in SEEDS)
                         for split in ("train", "id"))
    findings = {"status": "PASS" if oracle_control else "REJECT", "technical_validity": {
        "oracle_better_than_no_context_on_train_and_id": oracle_control,
        "all_learned_pipelines_complete": len(checkpoint_rows) == 10,
        "checkpoint_and_dataset_provenance_match": len({row["dataset_checksum"] for row in checkpoint_rows}) == 1,
        "five_paired_end_to_end_seeds": len({row["policy_seed"] for row in checkpoint_rows}) == 5},
        "independent_replicate": "end-to-end policy seed (with same-seed encoder for learned methods)",
        "bootstrap": {"seed": int(spec["analysis"]["analysis_seed"]), "resamples": 10000,
                      "label": "low-sample descriptive intervals"},
        "ood_role": "scientific outcome; descriptive only; never used for checkpoint or configuration selection",
        "ood_reporting": "left, right, near |angle|=0.8, far |angle|=1.0, and every goal angle retained",
        "pseudo_replication": "contexts and episodes are not treated as independent seeds"}
    (root / "primary_findings.json").write_text(json.dumps(findings, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = ["# PointRobot primary findings", "", f"Technical validity: **{findings['status']}**.", "",
        "Five paired end-to-end seeds are the independent replicates. Contexts and episodes are aggregated within seed and are never used as pseudo-replicates.", "",
        "Paired bootstrap 95% intervals use 10,000 deterministic resamples and are labeled low-sample descriptive intervals.", "",
        "OOD-left, OOD-right, near (|angle|=0.8), far (|angle|=1.0), and every individual angle are retained. OOD is the scientific outcome and had no selection role."]
    (root / "primary_findings.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    _plots(root, rows, by_seed, closure, near_far)
    return root


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/pointrobot_primary")
    parser.add_argument("--spec", default="configs/pointrobot_primary/spec.yaml")
    args = parser.parse_args(argv)
    output = analyze(args.results_dir, args.spec)
    print(f"wrote {len(OUTPUTS)} primary result-only analysis artifacts to {output}")


if __name__ == "__main__":
    main()
