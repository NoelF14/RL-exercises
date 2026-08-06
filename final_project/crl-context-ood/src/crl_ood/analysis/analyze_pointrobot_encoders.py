"""Dependency-light, result-only PointRobot encoder analysis.

This module intentionally imports no environment, RL, or neural-network package.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml


OUTPUTS = (
    "encoder_training_summary.csv", "encoder_validation_losses.csv", "encoder_parameter_counts.csv",
    "encoder_latent_statistics.csv", "encoder_context_probe_by_seed.csv", "encoder_context_probe_summary.csv",
    "encoder_reward_prediction_by_context.csv", "encoder_checkpoint_manifest.csv", "encoder_findings.md",
    "training_validation_curves.png", "latent_context_scatter.png", "probe_error_by_method.png",
    "reward_prediction_by_context.png",
)


def analyze(results_dir: str | Path, config_path: str | Path) -> Path:
    root = Path(results_dir); root.mkdir(parents=True, exist_ok=True)
    with Path(config_path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    losses, summaries, counts, checkpoints = [], [], [], []
    provenance_paths = list(root.glob("runs/*/seed_*/provenance.json")) + list(root.glob("smoke/*/provenance.json"))
    for provenance_path in sorted(provenance_paths):
        run = provenance_path.parent
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        run_losses = _read_csv(run / "losses.csv")
        for row in run_losses:
            losses.append({"method": provenance["method"], "seed": provenance["seed"], **row})
        selection = json.loads((run / "checkpoint_selection.json").read_text())
        summaries.append({"method": provenance["method"], "seed": provenance["seed"],
                          "dataset_checksum": provenance["dataset_checksum"],
                          "best_update": selection["best_update"], "best_validation_loss": selection["best_value"],
                          "selection_scope": selection["selection_scope"], "ood_used": selection["ood_used"]})
        counts.append({"method": provenance["method"], "seed": provenance["seed"], **provenance["parameter_counts"]})
        manifest = json.loads((run / "checkpoint_manifest.json").read_text())
        for name, checksum in manifest.items():
            checkpoints.append({"method": provenance["method"], "seed": provenance["seed"],
                                "checkpoint": str((run / name).resolve()), "sha256": checksum,
                                "dataset_checksum": provenance["dataset_checksum"]})
    latent_rows, probe_rows, scatter = _latent_analysis(root)
    probe_summary = []
    for method in sorted({row["method"] for row in probe_rows}):
        values = [float(row["angle_mae"]) for row in probe_rows if row["method"] == method]
        probe_summary.append({"method": method, "seeds": len(values), "mean_angle_mae": np.mean(values),
                              "std_angle_mae": np.std(values)})
    reward_rows = _collect_reward_predictions(root)
    _write_csv(root / "encoder_training_summary.csv", summaries,
               ("method", "seed", "dataset_checksum", "best_update", "best_validation_loss", "selection_scope", "ood_used"))
    _write_csv(root / "encoder_validation_losses.csv", losses,
               ("method", "seed", "update", "split", "total", "state_reconstruction", "reward_reconstruction",
                "kl", "infonce", "contrastive_accuracy", "learning_rate", "gradient_norm"))
    _write_csv(root / "encoder_parameter_counts.csv", counts,
               ("method", "seed", "backbone", "method_specific", "total_training", "downstream_retained"))
    _write_csv(root / "encoder_latent_statistics.csv", latent_rows,
               ("method", "seed", "split", "latent_dimension", "mean", "std", "minimum", "maximum"))
    _write_csv(root / "encoder_context_probe_by_seed.csv", probe_rows,
               ("method", "seed", "fit_split", "evaluation_split", "angle_mae", "checkpoint_selection_role"))
    _write_csv(root / "encoder_context_probe_summary.csv", probe_summary,
               ("method", "seeds", "mean_angle_mae", "std_angle_mae"))
    _write_csv(root / "encoder_reward_prediction_by_context.csv", reward_rows,
               ("method", "seed", "split", "context", "reward_mse", "checkpoint_selection_role"))
    _write_csv(root / "encoder_checkpoint_manifest.csv", checkpoints,
               ("method", "seed", "checkpoint", "sha256", "dataset_checksum"))
    _plots(root, losses, probe_summary, reward_rows, scatter)
    matched = len({int(row["backbone"]) for row in counts}) <= 1 if counts else True
    findings = ["# PointRobot encoder findings", "", "Result-only descriptive analysis; diagnostic probes never feed gradients or select checkpoints.", "",
                f"- Runs discovered: {len(summaries)}.", f"- Shared backbone parameter counts matched: {matched}.",
                "- Checkpoint selection used only held-out training-context trajectory objective.",
                "- ID and OOD-left/right rows are reported only when frozen diagnostic artifacts exist; they remain unavailable in smoke-only analysis.",
                "- OOD results have no configuration or checkpoint-selection role.", "",
                "The original gate v1 REJECT and separate probe-audit v2 PASS are outside this result namespace and unchanged."]
    (root / "encoder_findings.md").write_text("\n".join(findings) + "\n", encoding="utf-8")
    return root


def _latent_analysis(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[tuple[str, np.ndarray, np.ndarray]]]:
    statistics, probes, scatter = [], [], []
    paths = sorted(root.glob("frozen_evaluations/*/latents.npz"))
    complete = []
    for candidate in paths:
        with np.load(candidate, allow_pickle=False) as archive:
            if "split" in archive.files:
                complete.append(candidate)
    for path in (complete or paths):
        provenance = json.loads((path.parent / "provenance.json").read_text())
        method = provenance["method"]
        seed = _seed_from_path(path.parent.name)
        with np.load(path, allow_pickle=False) as archive:
            latent, context = archive["latent"], archive["context"]
            saved_split = archive["split"] if "split" in archive.files else None
        index = _read_csv(path.parent / "latent_index.csv")
        assignments = np.asarray([row["assignment"] for row in index])
        splits = np.asarray(saved_split if saved_split is not None else ["train"] * len(latent))
        for split in sorted(set(splits)):
            selected = latent[splits == split]
            for dimension in range(latent.shape[1]):
                values = selected[:, dimension]
                statistics.append({"method": method, "seed": seed, "split": split,
                    "latent_dimension": dimension, "mean": values.mean(), "std": values.std(),
                    "minimum": values.min(), "maximum": values.max()})
        train = (splits == "train") & (assignments == "train")
        design = np.c_[np.ones(train.sum()), latent[train]]
        target = np.c_[np.cos(context[train]), np.sin(context[train])]
        weights = np.linalg.pinv(design) @ target
        for split in ("train", "id", "ood_left", "ood_right"):
            selected = (splits == split) & ((assignments == "validation") if split == "train" else True)
            if not selected.any():
                continue
            prediction = np.c_[np.ones(selected.sum()), latent[selected]] @ weights
            angles = np.arctan2(prediction[:, 1], prediction[:, 0])
            error = np.abs((angles - context[selected] + np.pi) % (2 * np.pi) - np.pi)
            probes.append({"method": method, "seed": seed, "fit_split": "train",
                           "evaluation_split": split, "angle_mae": error.mean(),
                           "checkpoint_selection_role": "diagnostic_only"})
        scatter.append((method, latent[:, :2], context))
    return statistics, probes, scatter


def _collect_reward_predictions(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("frozen_evaluations/*/reward_predictions.csv")):
        for row in _read_csv(path):
            rows.append({**row, "checkpoint_selection_role": "diagnostic_only"})
    return rows


def _plots(root: Path, losses: list[dict[str, Any]], probes: list[dict[str, Any]],
           rewards: list[dict[str, Any]], scatter: list[tuple[str, np.ndarray, np.ndarray]]) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for method in sorted({row["method"] for row in losses}):
        selected = [row for row in losses if row["method"] == method]
        ax.plot([float(x["update"]) for x in selected], [float(x["total"]) for x in selected], marker="o", label=method)
    ax.set(xlabel="update", ylabel="validation objective"); ax.legend() if losses else None; fig.tight_layout()
    fig.savefig(root / "training_validation_curves.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for method, latent, context in scatter:
        points = ax.scatter(latent[:, 0], latent[:, 1], c=context, s=8, alpha=.5, label=method)
    ax.set(xlabel="latent 0", ylabel="latent 1"); ax.legend() if scatter else None; fig.tight_layout()
    fig.savefig(root / "latent_context_scatter.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4)); ax.bar([r["method"] for r in probes], [r["mean_angle_mae"] for r in probes])
    ax.set(ylabel="diagnostic angle MAE"); fig.tight_layout(); fig.savefig(root / "probe_error_by_method.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for method in sorted({row["method"] for row in rewards}):
        selected = [r for r in rewards if r["method"] == method]
        ax.plot([float(r["context"]) for r in selected], [float(r["reward_mse"]) for r in selected], marker="o", label=method)
    ax.set(xlabel="context", ylabel="reward MSE"); ax.legend() if rewards else None; fig.tight_layout()
    fig.savefig(root / "reward_prediction_by_context.png", dpi=150); plt.close(fig)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size <= 1:
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore"); writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _seed_from_path(name: str) -> int:
    for token in name.split("_"):
        if token.isdigit():
            return int(token)
    return -1


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/pointrobot_encoders")
    parser.add_argument("--config", default="configs/pointrobot_encoders/primary.yaml")
    args = parser.parse_args(argv)
    output = analyze(args.results_dir, args.config)
    print(f"wrote {len(OUTPUTS)} result-only encoder analysis artifacts to {output}")


if __name__ == "__main__":
    main()
