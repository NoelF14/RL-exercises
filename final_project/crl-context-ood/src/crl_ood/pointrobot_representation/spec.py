"""Torch/environment-free validation for the frozen representation protocol."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

METHODS = ("vae", "contrastive")
SEEDS = (0, 1, 2, 3, 4)
SPLITS = {
    "train": (-0.6, -0.3, 0.0, 0.3, 0.6),
    "id": (-0.45, -0.15, 0.15, 0.45),
    "ood_left": (-1.0, -0.8),
    "ood_right": (0.8, 1.0),
}
DATASET_SHA256 = "cb826e04b344eb875662b8775b89f9c60bdb9bae895f25a260d25ef422a589fa"
PRIMARY_SNAPSHOT = "6b9cd43da2cb9e276b6c772e1047435f142de1c5"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_spec(path: str | Path) -> dict[str, Any]:
    spec_path = Path(path)
    value = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("representation specification must be a mapping")
    if tuple(value.get("methods", ())) != METHODS or tuple(value.get("encoder_seeds", ())) != SEEDS:
        raise ValueError("methods and five encoder seeds are frozen")
    actual = {name: tuple(float(x) for x in value["contexts"].get(name, ())) for name in SPLITS}
    if actual != SPLITS or set(value["contexts"]) != set(SPLITS):
        raise ValueError("context splits are frozen and must be exact")
    if value["dataset"]["checksum"] != DATASET_SHA256:
        raise ValueError("primary dataset checksum is frozen")
    if value["experiment"]["authoritative_primary_source_snapshot"] != PRIMARY_SNAPSHOT:
        raise ValueError("authoritative primary source snapshot mismatch")
    if value["experiment"]["checkpoint_reselection"] != "forbidden":
        raise ValueError("checkpoint reselection is forbidden")
    history = value["history"]
    if (int(history["length"]) != 5 or int(history["transition_dimension"]) != 7
            or history["convention"] != "completed_transitions_max_0_t_minus_H_through_t_minus_1"
            or history["current_state_outside_latent"] is not True
            or history["empty_history_latent"] != "exact_zero_vector"):
        raise ValueError("leakage-safe H=5 completed-transition history is frozen")
    if int(value["latent"]["dimension"]) != 8 or value["latent"]["inference"] != "deterministic_frozen":
        raise ValueError("latent extraction must be deterministic frozen inference with dimension 8")
    diagnostic = value["diagnostic_trajectories"]
    n = int(diagnostic["trajectories_per_context"])
    expected_seeds = tuple(range(int(diagnostic["trajectory_seed_offset"]), int(diagnostic["trajectory_seed_offset"]) + n))
    if tuple(int(x) for x in diagnostic["trajectory_seeds"]) != expected_seeds:
        raise ValueError("diagnostic trajectory seeds are not the frozen consecutive sequence")
    if diagnostic["action_sequences_shared_across_methods_seeds_and_contexts"] is not True:
        raise ValueError("diagnostic action trajectories must be shared everywhere")
    if diagnostic["ppo_trajectories_forbidden"] is not True:
        raise ValueError("PPO trajectories are forbidden for primary representation probes")
    probe, pca = value["probe"], value["pca"]
    if probe["fit_split"] != "train" or tuple(probe["evaluation_splits"]) != tuple(SPLITS):
        raise ValueError("probe fitting is train-context only")
    if probe["state_only_features"] != ["current_state_x", "current_state_y"] or not probe["state_only_history_forbidden"]:
        raise ValueError("state-only baseline may contain current state only")
    if pca["fit_split"] != "train" or pca["standardization"] != "none" or pca["cross_seed_alignment"] != "forbidden":
        raise ValueError("PCA is train-only, unstandardized, and never cross-seed aligned")
    if int(pca["main_visualization_seed"]) != 0:
        raise ValueError("seed 0 is prespecified for the compact PCA figure")
    value["_spec_path"] = str(spec_path.resolve())
    value["_configuration_checksum"] = sha256_file(spec_path)
    return value


@dataclass(frozen=True)
class CheckpointJob:
    method: str
    encoder_seed: int
    checkpoint: Path
    checkpoint_sha256: str
    output_dir: Path


def build_checkpoint_jobs(spec: dict[str, Any], root: str | Path) -> list[CheckpointJob]:
    base = Path(root)
    rule = str(spec["checkpoints"]["rule"])
    jobs = []
    for method in METHODS:
        for seed in SEEDS:
            relative = rule.format(method=method, seed=seed)
            expected = str(spec["checkpoints"]["sha256"][method][seed])
            jobs.append(CheckpointJob(method, seed, base / relative, expected,
                base / spec["experiment"]["results_dir"] / "evaluations" / method / f"seed_{seed}"))
    identities = {(job.method, job.encoder_seed) for job in jobs}
    paths = {job.checkpoint.resolve() for job in jobs}
    if len(jobs) != 10 or len(identities) != 10 or len(paths) != 10:
        raise ValueError("exactly ten unique checkpoint jobs are required")
    return jobs


def validate_primary_provenance(spec: dict[str, Any], root: str | Path) -> list[dict[str, Any]]:
    base = Path(root)
    checksum_path = base / spec["dataset"]["checksum_artifact"]
    metadata_path = base / spec["dataset"]["metadata_artifact"]
    checksum = checksum_path.read_text(encoding="ascii").strip()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if checksum != DATASET_SHA256 or metadata.get("dataset_checksum") != DATASET_SHA256:
        raise ValueError("primary dataset checksum validation failed")
    if metadata.get("source_git_commit") != spec["experiment"]["primary_execution_source_commit"]:
        raise ValueError("primary dataset source provenance mismatch")
    manifest_path = base / spec["checkpoints"]["provenance_manifest"]
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        primary_rows = {(row["method"], int(row["encoder_seed"])): row for row in csv.DictReader(handle)}
    jobs = build_checkpoint_jobs(spec, base)
    if set(primary_rows) != {(job.method, job.encoder_seed) for job in jobs}:
        raise ValueError("primary checkpoint provenance must contain exactly the selected ten checkpoints")
    rows = []
    for job in jobs:
        if not job.checkpoint.is_file():
            raise FileNotFoundError(job.checkpoint)
        actual = sha256_file(job.checkpoint)
        primary = primary_rows[(job.method, job.encoder_seed)]
        run = job.checkpoint.parent
        run_manifest = json.loads((run / "checkpoint_manifest.json").read_text(encoding="utf-8"))
        provenance = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
        selection = json.loads((run / "checkpoint_selection.json").read_text(encoding="utf-8"))
        resolved = yaml.safe_load((run / "resolved_config.yaml").read_text(encoding="utf-8"))
        hashes = {actual, job.checkpoint_sha256, primary["encoder_checkpoint_sha256"], run_manifest.get("best.pt")}
        if len(hashes) != 1:
            raise ValueError(f"checkpoint SHA validation failed for {job.method} seed {job.encoder_seed}")
        if (provenance.get("method") != job.method or int(provenance.get("seed", -1)) != job.encoder_seed
                or provenance.get("dataset_checksum") != DATASET_SHA256
                or provenance.get("source_commit") != spec["experiment"]["primary_execution_source_commit"]):
            raise ValueError("encoder run source/dataset/method/seed provenance mismatch")
        if (selection.get("selection_scope") != "held-out training-context trajectories only"
                or selection.get("ood_used") is not False):
            raise ValueError("checkpoint selection provenance violates the no-reselection rule")
        if primary["dataset_checksum"] != DATASET_SHA256:
            raise ValueError("primary checkpoint manifest dataset checksum mismatch")
        if (Path(primary["encoder_checkpoint_path"]).resolve() != job.checkpoint.resolve()
                or primary["source_commit"] != spec["experiment"]["primary_execution_source_commit"]
                or primary["configuration_checksum"] != spec["experiment"]["primary_configuration_checksum"]):
            raise ValueError("primary checkpoint path/source/configuration provenance mismatch")
        encoder = resolved.get("encoder", {})
        if (int(encoder.get("history_length", -1)) != 5 or int(encoder.get("transition_dim", -1)) != 7
                or int(encoder.get("latent_dim", -1)) != 8):
            raise ValueError("trained checkpoint configuration is inconsistent with frozen representation dimensions")
        rows.append({"method": job.method, "encoder_seed": job.encoder_seed,
            "checkpoint_path": str(job.checkpoint.resolve()), "checkpoint_sha256": actual,
            "dataset_checksum": DATASET_SHA256,
            "primary_execution_source_commit": provenance["source_commit"],
            "authoritative_primary_source_snapshot": PRIMARY_SNAPSHOT,
            "configuration_checksum": spec["_configuration_checksum"],
            "selection_role": "diagnostic_only", "checkpoint_reselected": False})
    return rows


def dry_run_lines(spec: dict[str, Any], root: str | Path) -> list[str]:
    jobs = build_checkpoint_jobs(spec, root)
    lines = ["FROZEN REPRESENTATION PLAN: 10 checkpoints; no Torch/environment loading; no execution"]
    for number, job in enumerate(jobs, 1):
        lines.append(f"{number:02d} {job.method} seed={job.encoder_seed} checkpoint={job.checkpoint} sha256={job.checkpoint_sha256} output={job.output_dir}")
    output = Path(root) / spec["experiment"]["results_dir"]
    lines.append(f"evaluation_manifest={output / spec['analysis']['evaluation_manifest']}")
    lines.append(f"analysis_output={output}")
    return lines
