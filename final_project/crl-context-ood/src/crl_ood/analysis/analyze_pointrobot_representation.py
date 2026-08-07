"""Result-only PointRobot representation analysis.

This module intentionally imports no Torch, Gym/Gymnasium, CARL, or
Stable-Baselines3 code. It reads only frozen CSV/JSON/YAML/NPY/NPZ artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

from crl_ood.pointrobot_representation.manifest import verify_manifest, write_manifest
from crl_ood.pointrobot_representation.spec import METHODS, SEEDS, SPLITS, load_spec

CSV_OUTPUTS = (
    "representation_latent_index.csv", "representation_probe_predictions.csv",
    "representation_probe_by_angle.csv", "representation_probe_by_seed.csv",
    "representation_probe_summary.csv", "representation_state_only_probe.csv",
    "representation_pca_coordinates.csv", "representation_pca_summary.csv",
    "representation_checkpoint_manifest.csv", "representation_control_by_seed.csv",
)
FIGURES = (
    "probe_mae_by_split", "probe_mae_by_angle", "latent_pca_seed_0", "latent_pca_all_seeds",
    "probe_vs_return_id", "probe_vs_return_ood_left", "probe_vs_return_ood_right",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() != ".csv":
        raise ValueError("result-only analysis accepts persisted CSV inputs")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: Iterable[str] | None = None) -> None:
    fields = list(columns or (rows[0] if rows else ()))
    if not fields:
        raise ValueError(f"no output schema for {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def _mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(np.mean(array))


def _std(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(np.std(array, ddof=1)) if len(array) > 1 else math.nan


def _collect(results: Path, manifest_paths: list[Path], spec: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    manifest_set = {path.resolve() for path in manifest_paths}
    collections: dict[str, list[dict[str, Any]]] = {
        "latent_samples.csv": [], "probe_predictions.csv": [], "probe_by_angle.csv": [],
        "probe_by_seed.csv": [], "state_only_probe.csv": [], "pca_coordinates.csv": [],
    }
    checkpoint_rows, pca_rows = [], []
    identities = set()
    for method in METHODS:
        for seed in SEEDS:
            run = results / "evaluations" / method / f"seed_{seed}"
            provenance_path = run / "provenance.json"
            if provenance_path.resolve() not in manifest_set:
                raise ValueError(f"evaluation manifest omits {provenance_path.relative_to(results)}")
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            identity = (provenance.get("method"), int(provenance.get("encoder_seed", -1)))
            if identity != (method, seed) or identity in identities:
                raise ValueError("evaluation provenance must contain exactly ten unique method/seed jobs")
            identities.add(identity)
            if (provenance.get("dataset_checksum") != spec["dataset"]["checksum"]
                    or provenance.get("configuration_checksum") != spec["_configuration_checksum"]
                    or provenance.get("selection_role") != "diagnostic_only"):
                raise ValueError("evaluation provenance is inconsistent with the frozen specification")
            checkpoint_rows.append({key: provenance[key] for key in (
                "method", "encoder_seed", "checkpoint_path", "checkpoint_sha256", "dataset_checksum",
                "primary_execution_source_commit", "authoritative_primary_source_snapshot",
                "configuration_checksum", "selection_role", "checkpoint_reselected")})
            for name in collections:
                path = run / name
                if path.resolve() not in manifest_set:
                    raise ValueError(f"evaluation manifest omits {path.relative_to(results)}")
                rows = _read_csv(path)
                if any(row.get("method") != method or int(row.get("encoder_seed", -1)) != seed for row in rows):
                    raise ValueError(f"method/seed mismatch in {path}")
                collections[name].extend(rows)
            pca_path = run / "pca_model.npz"
            if pca_path.resolve() not in manifest_set:
                raise ValueError("evaluation manifest omits PCA model")
            with np.load(pca_path, allow_pickle=False) as archive:
                if str(archive["fit_split"]) != "train" or str(archive["standardization"]) != "none":
                    raise ValueError("PCA model was not fitted under the frozen train-only rule")
                explained = archive["explained_variance"]; ratios = archive["explained_variance_ratio"]
                components = archive["components"]
                if components.shape != (2, 8):
                    raise ValueError("each independent checkpoint PCA must retain two components over 8-D latents")
                for component in range(2):
                    pca_rows.append({"method": method, "encoder_seed": seed, "component": component + 1,
                        "explained_variance": float(explained[component]),
                        "explained_variance_ratio": float(ratios[component]), "fit_split": "train",
                        "standardization": "none", "cross_seed_alignment": False,
                        **{f"loading_z_{index}": float(components[component, index]) for index in range(8)}})
    if identities != {(method, seed) for method in METHODS for seed in SEEDS}:
        raise ValueError("exactly 5 VAE and 5 contrastive encoder seeds are required")
    collections["checkpoint_manifest"] = checkpoint_rows
    collections["pca_summary"] = pca_rows
    return collections


def _validate_samples(collections: dict[str, list[dict[str, Any]]]) -> None:
    latent = collections["latent_samples.csv"]
    expected_contexts = {(split, angle) for split, angles in SPLITS.items() for angle in angles}
    reference_trajectories = None
    for method in METHODS:
        method_seeds = {int(row["encoder_seed"]) for row in latent if row["method"] == method}
        if method_seeds != set(SEEDS):
            raise ValueError("five encoder seeds, not contexts or samples, must be the replicates")
        for seed in SEEDS:
            rows = [row for row in latent if row["method"] == method and int(row["encoder_seed"]) == seed]
            actual = {(row["split"], float(row["goal_angle"])) for row in rows}
            if actual != expected_contexts:
                raise ValueError("signed contexts and directional OOD splits were not preserved")
            if any(set(f"z_{index}" for index in range(8)) - set(row) for row in rows):
                raise ValueError("latent dimensionality must be exactly 8")
            trajectories = {(row["split"], float(row["goal_angle"]), int(row["trajectory_id"]),
                int(row["trajectory_seed"]), int(row["timestep"])) for row in rows}
            reference_trajectories = trajectories if reference_trajectories is None else reference_trajectories
            if trajectories != reference_trajectories:
                raise ValueError("diagnostic trajectories differ across methods or encoder seeds")
    predictions = collections["probe_predictions.csv"]
    if any(row["probe_fit_split"] != "train" for row in predictions):
        raise ValueError("probe fitting must use training contexts only")
    state = collections["state_only_probe.csv"]
    if any(row["features"] != "current_state_only" or row["contains_history"].lower() != "false" for row in state):
        raise ValueError("state-only baseline contains history information")
    pca = collections["pca_coordinates.csv"]
    if any(row["pca_fit_split"] != "train" or row["standardization"] != "none"
           or row["cross_seed_alignment"].lower() != "false" for row in pca):
        raise ValueError("PCA train-only/no-alignment provenance failed")


def _annotate_state_only(rows: list[dict[str, Any]]) -> None:
    """Keep per-sample baseline predictions while attaching angle/split/seed summaries."""
    split_seed = {}
    angle_seed = {}
    for method in METHODS:
        for seed in SEEDS:
            for split, angles in SPLITS.items():
                selected = [row for row in rows if row["method"] == method
                    and int(row["encoder_seed"]) == seed and row["split"] == split]
                split_seed[(method, seed, split)] = _mean(float(row["circular_absolute_angle_error"]) for row in selected)
                for angle in angles:
                    by_angle = [row for row in selected if np.isclose(float(row["goal_angle"]), angle)]
                    angle_seed[(method, seed, split, angle)] = _mean(
                        float(row["circular_absolute_angle_error"]) for row in by_angle)
    for row in rows:
        method, seed, split, angle = row["method"], int(row["encoder_seed"]), row["split"], float(row["goal_angle"])
        seed_values = [split_seed[(method, item, split)] for item in SEEDS]
        row["angle_circular_angle_mae"] = angle_seed[(method, seed, split, angle)]
        row["seed_split_circular_angle_mae"] = split_seed[(method, seed, split)]
        row["five_seed_mean_split_mae"] = _mean(seed_values)
        row["five_seed_std_split_mae"] = _std(seed_values)
        row["independent_replicate"] = "encoder_seed"
        row["encoder_seed_count"] = 5


def _probe_summary(by_seed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for method in METHODS:
        for split in SPLITS:
            selected = [row for row in by_seed if row["method"] == method and row["split"] == split]
            seeds = {int(row["encoder_seed"]) for row in selected}
            if seeds != set(SEEDS) or len(selected) != 5:
                raise ValueError("probe summary requires exactly five encoder-seed replicates")
            values = [float(row["circular_angle_mae"]) for row in selected]
            rows.append({"method": method, "split": split, "independent_replicate": "encoder_seed",
                "encoder_seed_count": 5, "context_count_is_replicate": False, "sample_count_is_replicate": False,
                "seed_values": json.dumps(values, separators=(",", ":")),
                "mean_circular_angle_mae": _mean(values), "std_circular_angle_mae": _std(values)})
    return rows


def _control_join(by_seed: list[dict[str, Any]], primary_dir: Path) -> list[dict[str, Any]]:
    primary = _read_csv(primary_dir / "primary_summary_by_seed.csv")
    persisted_closure = _read_csv(primary_dir / "primary_oracle_gap_closure.csv")
    closure_lookup = {(row["method"], int(row["policy_seed"]), row["split"]): float(row["oracle_gap_closure"])
        for row in persisted_closure}
    lookup = {(row["method"], int(row["policy_seed"]), row["split"]): float(row["mean_return"]) for row in primary}
    rows = []
    for probe in by_seed:
        split = probe["split"]
        if split not in {"id", "ood_left", "ood_right"}:
            continue
        method, seed = probe["method"], int(probe["encoder_seed"])
        learned = lookup[(method, seed, split)]; baseline = lookup[("no_context", seed, split)]
        oracle = lookup[("oracle", seed, split)]
        closure = closure_lookup.get((method, seed, split))
        if closure is None and oracle != baseline:
            closure = (learned - baseline) / (oracle - baseline)
        rows.append({"method": method, "encoder_seed": seed, "split": split,
            "probe_circular_angle_mae": float(probe["circular_angle_mae"]), "mean_return": learned,
            "oracle_gap_closure": closure if closure is not None else math.nan,
            "relationship_role": "descriptive_only_n_equals_5", "inferential_test": "not_performed"})
    return rows


def _save_figure(fig: Any, results: Path, stem: str) -> None:
    fig.savefig(results / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(results / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _figures(results: Path, by_seed: list[dict[str, Any]], by_angle: list[dict[str, Any]],
             pca: list[dict[str, Any]], control: list[dict[str, Any]]) -> None:
    colors = {"vae": "#3366aa", "contrastive": "#cc6633"}
    split_names = list(SPLITS)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(split_names)); width = .36
    for offset, method in enumerate(METHODS):
        means, stds = [], []
        for split in split_names:
            values = [float(row["circular_angle_mae"]) for row in by_seed if row["method"] == method and row["split"] == split]
            means.append(_mean(values)); stds.append(_std(values))
        ax.bar(x + (offset - .5) * width, means, width, yerr=stds, label=method, color=colors[method], capsize=3)
    ax.set_xticks(x, [name.replace("_", "\n") for name in split_names]); ax.set_ylabel("Circular angle MAE (rad)")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.2); fig.tight_layout(); _save_figure(fig, results, "probe_mae_by_split")

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for method in METHODS:
        angles = sorted({float(row["goal_angle"]) for row in by_angle if row["method"] == method})
        values = [_mean(float(row["circular_angle_mae"]) for row in by_angle
            if row["method"] == method and np.isclose(float(row["goal_angle"]), angle)) for angle in angles]
        ax.plot(angles, values, marker="o", label=method, color=colors[method])
    ax.axvspan(-1, -.8, color="#999999", alpha=.12); ax.axvspan(.8, 1, color="#999999", alpha=.12)
    ax.set_xlabel("Signed goal angle (rad)"); ax.set_ylabel("Mean circular angle MAE (rad)")
    ax.legend(frameon=False); ax.grid(alpha=.2); fig.tight_layout(); _save_figure(fig, results, "probe_mae_by_angle")

    context_colors = plt.get_cmap("coolwarm")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharex=False, sharey=False)
    for ax, method in zip(axes, METHODS, strict=True):
        rows = [row for row in pca if row["method"] == method and int(row["encoder_seed"]) == 0]
        ax.scatter([float(row["pc_1"]) for row in rows], [float(row["pc_2"]) for row in rows],
            c=[float(row["goal_angle"]) for row in rows], cmap=context_colors, s=3, alpha=.22, rasterized=True)
        ax.set_title(f"{method}, seed 0 (prespecified)"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.tight_layout(); _save_figure(fig, results, "latent_pca_seed_0")

    fig, axes = plt.subplots(2, 5, figsize=(12, 5.2))
    for row_index, method in enumerate(METHODS):
        for seed in SEEDS:
            ax = axes[row_index, seed]; rows = [row for row in pca if row["method"] == method and int(row["encoder_seed"]) == seed]
            ax.scatter([float(row["pc_1"]) for row in rows], [float(row["pc_2"]) for row in rows],
                c=[float(row["goal_angle"]) for row in rows], cmap=context_colors, s=1.5, alpha=.18, rasterized=True)
            ax.set_title(f"{method} s{seed}", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Independent train-fitted PCA coordinates (no cross-seed alignment)", fontsize=11)
    fig.tight_layout(); _save_figure(fig, results, "latent_pca_all_seeds")

    for split in ("id", "ood_left", "ood_right"):
        fig, ax = plt.subplots(figsize=(4.4, 3.6))
        for method in METHODS:
            rows = [row for row in control if row["method"] == method and row["split"] == split]
            ax.scatter([float(row["probe_circular_angle_mae"]) for row in rows], [float(row["mean_return"]) for row in rows],
                label=method, color=colors[method], s=38)
            for row in rows:
                ax.annotate(str(row["encoder_seed"]), (float(row["probe_circular_angle_mae"]), float(row["mean_return"])),
                    xytext=(3, 2), textcoords="offset points", fontsize=7)
        ax.set_xlabel("Probe circular angle MAE (rad)"); ax.set_ylabel("Primary mean return")
        ax.set_title(split.replace("_", " ")); ax.legend(frameon=False); ax.grid(alpha=.2); fig.tight_layout()
        _save_figure(fig, results, f"probe_vs_return_{split}")


def analyze(results_dir: str | Path, spec_path: str | Path, primary_dir: str | Path) -> Path:
    results = Path(results_dir).resolve(); spec = load_spec(spec_path)
    evaluation_manifest = results / spec["analysis"]["evaluation_manifest"]
    manifest_paths = verify_manifest(results, evaluation_manifest)
    collections = _collect(results, manifest_paths, spec); _validate_samples(collections)
    _annotate_state_only(collections["state_only_probe.csv"])
    by_seed = collections["probe_by_seed.csv"]
    summary = _probe_summary(by_seed); control = _control_join(by_seed, Path(primary_dir))
    outputs = {
        "representation_latent_index.csv": collections["latent_samples.csv"],
        "representation_probe_predictions.csv": collections["probe_predictions.csv"],
        "representation_probe_by_angle.csv": collections["probe_by_angle.csv"],
        "representation_probe_by_seed.csv": by_seed,
        "representation_probe_summary.csv": summary,
        "representation_state_only_probe.csv": collections["state_only_probe.csv"],
        "representation_pca_coordinates.csv": collections["pca_coordinates.csv"],
        "representation_pca_summary.csv": collections["pca_summary"],
        "representation_checkpoint_manifest.csv": collections["checkpoint_manifest"],
        "representation_control_by_seed.csv": control,
    }
    analysis_files = []
    for name, rows in outputs.items():
        path = results / name; _write_csv(path, rows); analysis_files.append(path)
    findings = {"status": "descriptive_representation_analysis_complete", "selection_role": "diagnostic_only",
        "independent_replicate": "encoder_seed", "encoder_seed_count_per_method": 5,
        "contexts_and_samples_counted_as_seeds": False, "inferential_correlation_tests": "not_performed_n_equals_5",
        "probe_fit_scope": "train contexts only", "pca_fit_scope": "train contexts only",
        "pca_cross_seed_alignment": False, "main_pca_seed": 0,
        "probe_summary": summary}
    findings_json = results / "representation_findings.json"
    findings_json.write_text(json.dumps(findings, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    findings_md = results / "representation_findings.md"
    findings_md.write_text("# PointRobot representation findings\n\n"
        "This file is generated only after the frozen evaluation manifest verifies. Results are diagnostic and were not used for checkpoint selection. "
        "Linear probes and PCA are fitted independently per checkpoint using train-context samples only. Five encoder seeds are the independent replicates; contexts and samples are not seeds. "
        "Control relationships are descriptive only, with no inferential correlation tests at n=5. Seed 0 is the prespecified compact PCA visualization and all seeds remain in the supplemental figure.\n",
        encoding="utf-8")
    analysis_files.extend((findings_json, findings_md))
    _figures(results, by_seed, collections["probe_by_angle.csv"], collections["pca_coordinates.csv"], control)
    analysis_files.extend(results / f"{stem}.{suffix}" for stem in FIGURES for suffix in ("png", "pdf"))
    analysis_manifest = results / spec["analysis"]["analysis_manifest"]
    write_manifest(results, analysis_files, analysis_manifest)
    final_manifest = results / spec["analysis"]["final_manifest"]
    write_manifest(results, [evaluation_manifest, analysis_manifest, *analysis_files], final_manifest)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/pointrobot_representation")
    parser.add_argument("--spec", default="configs/pointrobot_representation/spec.yaml")
    parser.add_argument("--primary-dir", default="results/pointrobot_primary")
    args = parser.parse_args()
    print(analyze(args.results_dir, args.spec, args.primary_dir))


if __name__ == "__main__":
    main()
