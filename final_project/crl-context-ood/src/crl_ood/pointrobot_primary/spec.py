"""Pure validation and provenance helpers for the frozen primary specification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

METHODS = ("no_context", "oracle", "vae", "contrastive")
LEARNED_METHODS = ("vae", "contrastive")
SEEDS = (0, 1, 2, 3, 4)
SPLITS = {
    "train": (-0.6, -0.3, 0.0, 0.3, 0.6),
    "id": (-0.45, -0.15, 0.15, 0.45),
    "ood_left": (-1.0, -0.8),
    "ood_right": (0.8, 1.0),
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_spec(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("primary specification must be a mapping")
    required = {"format_version", "experiment", "dataset", "contexts", "encoder", "policy", "evaluation", "analysis"}
    missing = required - set(value)
    if missing:
        raise ValueError(f"primary specification is incomplete: missing {sorted(missing)}")
    if tuple(value["policy"]["methods"]) != METHODS or tuple(value["policy"]["seeds"]) != SEEDS:
        raise ValueError("primary policy methods/seeds are frozen")
    if tuple(value["encoder"]["methods"]) != LEARNED_METHODS or tuple(value["encoder"]["seeds"]) != SEEDS:
        raise ValueError("primary encoder methods/seeds are frozen")
    actual_splits = {name: tuple(float(x) for x in value["contexts"][name]) for name in SPLITS}
    if actual_splits != SPLITS:
        raise ValueError("primary context splits are frozen")
    architecture = value["encoder"]["architecture"]
    expected_architecture = {"num_layers": 1, "hidden_size": 64, "latent_size": 8,
                             "retained_parameter_count": 14536, "history_length": 5, "future_horizon": 5}
    if any(int(architecture[key]) != expected for key, expected in expected_architecture.items()):
        raise ValueError("primary encoder architecture is not the matched frozen architecture")
    if int(value["encoder"]["optimizer"]["max_updates"]) != 20000:
        raise ValueError("primary encoder update budget must be 20000")
    if int(value["policy"]["requested_timesteps"]) != 200000 or int(value["policy"]["expected_job_count"]) != 20:
        raise ValueError("primary downstream budget/job count is frozen")
    if value["evaluation"]["ood_role"] != "descriptive_scientific_evaluation_only":
        raise ValueError("OOD must be descriptive/scientific only")
    if not value["dataset"].get("checksum_artifact") or not value["dataset"].get("expected_checksum_field"):
        raise ValueError("dataset checksum provenance fields are required")
    value["_spec_path"] = str(config_path.resolve())
    value["_configuration_checksum"] = sha256_file(config_path)
    return value


@dataclass(frozen=True)
class EncoderJob:
    method: str
    encoder_seed: int
    dataset_dir: Path
    output_dir: Path
    dataset_checksum: str


@dataclass(frozen=True)
class PrimaryJob:
    method: str
    policy_seed: int
    encoder_seed: int | None
    checkpoint: Path | None
    checkpoint_sha256: str | None
    dataset_checksum: str
    requested_timesteps: int
    rollout_quantum: int
    output_dir: Path


def read_dataset_checksum(spec: dict[str, Any], root: Path) -> str:
    path = root / spec["dataset"]["checksum_artifact"]
    if not path.is_file():
        raise RuntimeError(f"primary matrix unavailable: missing full dataset checksum artifact {path}")
    value = path.read_text(encoding="ascii").strip()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("full dataset checksum artifact is invalid")
    metadata_path = path.with_name("dataset.json")
    if not metadata_path.is_file():
        raise RuntimeError("primary matrix unavailable: full dataset metadata is missing")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get(spec["dataset"]["expected_checksum_field"]) != value:
        raise ValueError("dataset checksum and metadata disagree")
    return value


def build_encoder_jobs(spec: dict[str, Any], root: Path) -> list[EncoderJob]:
    checksum = read_dataset_checksum(spec, root)
    dataset_dir = (root / spec["dataset"]["checksum_artifact"]).parent
    output_root = root / spec["experiment"]["results_dir"] / "encoders"
    jobs = [EncoderJob(method, seed, dataset_dir, output_root / method / f"seed_{seed}", checksum)
            for method in LEARNED_METHODS for seed in SEEDS]
    if len(jobs) != 10 or len({(job.method, job.encoder_seed) for job in jobs}) != 10:
        raise AssertionError("encoder matrix must contain exactly ten unique jobs")
    return jobs


def validate_encoder_runs(spec: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    rows = []
    for job in build_encoder_jobs(spec, root):
        provenance_path = job.output_dir / "provenance.json"
        selection_path = job.output_dir / "checkpoint_selection.json"
        manifest_path = job.output_dir / "checkpoint_manifest.json"
        checkpoint = job.output_dir / "best.pt"
        required = (provenance_path, selection_path, manifest_path, checkpoint, job.output_dir / "run.log")
        if not all(path.is_file() for path in required):
            raise RuntimeError(f"primary matrix unavailable: incomplete encoder run {job.method} seed {job.encoder_seed}")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual_hash = sha256_file(checkpoint)
        if provenance.get("method") != job.method or int(provenance.get("seed", -1)) != job.encoder_seed:
            raise ValueError("encoder method/seed provenance mismatch")
        if provenance.get("dataset_checksum") != job.dataset_checksum:
            raise ValueError("all encoder runs must reference the immutable full-dataset checksum")
        if provenance.get("source_commit") != spec["experiment"]["source_commit"]:
            raise ValueError("encoder source commit mismatch")
        counts = provenance.get("parameter_counts", {})
        if int(counts.get("downstream_retained", -1)) != 14536:
            raise ValueError("encoder retained parameter count mismatch")
        if selection.get("selection_scope") != "held-out training-context trajectories only" or selection.get("ood_used") is not False:
            raise ValueError("encoder checkpoint selection must use training contexts only")
        if manifest.get("best.pt") != actual_hash:
            raise ValueError("encoder checkpoint hash verification failed")
        completion = job.output_dir.joinpath("run.log").read_text()
        if not completion.startswith("COMPLETE") or "updates=20000" not in completion:
            raise RuntimeError("primary matrix unavailable: encoder run is not complete")
        rows.append({"method": job.method, "encoder_seed": job.encoder_seed,
                     "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": actual_hash,
                     "dataset_checksum": job.dataset_checksum})
    return rows


def build_downstream_jobs(spec: dict[str, Any], root: Path) -> list[PrimaryJob]:
    checksum = read_dataset_checksum(spec, root)
    checkpoints = {(row["method"], row["encoder_seed"]): row for row in validate_encoder_runs(spec, root)}
    output_root = root / spec["experiment"]["results_dir"] / "downstream"
    jobs = []
    for method in METHODS:
        for seed in SEEDS:
            row = checkpoints.get((method, seed)) if method in LEARNED_METHODS else None
            jobs.append(PrimaryJob(method, seed, seed if row else None,
                Path(row["checkpoint"]) if row else None, row["checkpoint_sha256"] if row else None,
                checksum, int(spec["policy"]["requested_timesteps"]), int(spec["policy"]["ppo"]["n_steps"]),
                output_root / method / f"seed_{seed}"))
    if len(jobs) != 20 or len({(job.method, job.policy_seed) for job in jobs}) != 20:
        raise AssertionError("downstream matrix must contain exactly twenty unique jobs")
    learned = [job for job in jobs if job.method in LEARNED_METHODS]
    if any(job.encoder_seed != job.policy_seed or job.checkpoint is None for job in learned):
        raise ValueError("learned policy seed s must use its method's encoder seed s checkpoint")
    for method in LEARNED_METHODS:
        if len({job.checkpoint for job in learned if job.method == method}) != 5:
            raise ValueError("one-checkpoint-for-all-seeds mapping is forbidden")
    return jobs


def validate_timestep_budget(requested: int, actual: int, rollout_quantum: int) -> None:
    expected = ((int(requested) + int(rollout_quantum) - 1) // int(rollout_quantum)) * int(rollout_quantum)
    if actual != expected:
        raise ValueError(f"actual complete-rollout timesteps {actual} != expected {expected}")
