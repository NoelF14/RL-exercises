"""Construct and validate the separate Phase 0 diagnostic job matrix."""

from __future__ import annotations

import copy
import csv
import json
import zipfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from crl_ood.utils.paths import run_directory


EXPECTED_EXPERIMENTS = {
    "default_100k": ("fixed", 1.0, 100_000, ("hidden",)),
    "default_300k": ("fixed", 1.0, 300_000, ("hidden",)),
    "specialist_low_300k": ("fixed", 0.8, 300_000, ("hidden",)),
    "specialist_center_300k": ("fixed", 1.0, 300_000, ("hidden",)),
    "specialist_high_300k": ("fixed", 1.2, 300_000, ("hidden",)),
    "contextual_300k": ("contextual", None, 300_000, ("hidden", "oracle")),
}

CSV_REQUIREMENTS = {
    "contexts.csv": {"run_id", "method", "context_feature", "split", "context_value"},
    "evaluation_plan.csv": {
        "run_id", "method", "seed", "context_feature", "split", "episode_seed"
    },
    "training_metrics.csv": {
        "run_id", "method", "seed", "context_feature", "environment_steps",
        "episode_return"
    },
    "episode_returns.csv": {
        "run_id", "method", "seed", "context_feature", "context_value", "split",
        "episode_index", "episode_seed", "return"
    },
    "context_returns.csv": {
        "run_id", "method", "seed", "context_feature", "context_value", "split",
        "mean_return"
    },
}
NONEMPTY_ARTIFACTS = (
    "metadata.json",
    "contexts.yaml",
    "model.zip",
    "sb3_monitor.csv",
    "sb3_logs/progress.csv",
)


@dataclass(frozen=True)
class DiagnosticJob:
    """One independently trainable and evaluable diagnostic run."""

    job_id: str
    config_path: Path
    config: dict[str, Any]
    experiment: str
    feature: str
    mode: str
    seed: int
    total_timesteps: int
    output_dir: Path


class RunState(str, Enum):
    PENDING = "pending"
    COMPLETE = "complete"
    PARTIAL = "partial"


@dataclass(frozen=True)
class RunStatus:
    state: RunState
    detail: str


def build_diagnostic_matrix(matrix_path: str | Path) -> list[DiagnosticJob]:
    """Load, validate, and deterministically expand all 14 diagnostic jobs."""
    matrix_path = Path(matrix_path).resolve()
    manifest = _load_yaml(matrix_path)
    phase0_path = _relative_to(matrix_path.parent, manifest["phase0_config"])
    phase0 = _load_yaml(phase0_path)
    config_paths = [_relative_to(matrix_path.parent, item) for item in manifest["configs"]]
    configs = [(path, _load_yaml(path)) for path in config_paths]
    _validate_configs(phase0_path, phase0, configs)

    jobs = []
    for config_path, config in configs:
        experiment = str(config["experiment"]["name"])
        feature = str(config["experiment"]["context_features"][0])
        total_timesteps = int(config["training"]["total_timesteps"])
        for seed in config["experiment"]["seeds"]:
            for mode in config["experiment"]["modes"]:
                seed = int(seed)
                mode = str(mode)
                job_id = f"{experiment}__{feature}__{mode}__seed_{seed}"
                jobs.append(
                    DiagnosticJob(
                        job_id=job_id,
                        config_path=config_path,
                        config=config,
                        experiment=experiment,
                        feature=feature,
                        mode=mode,
                        seed=seed,
                        total_timesteps=total_timesteps,
                        output_dir=run_directory(config, feature, mode, seed),
                    )
                )
    if len(jobs) != 14:
        raise ValueError(f"Diagnostic matrix must contain 14 jobs, found {len(jobs)}")
    job_ids = [job.job_id for job in jobs]
    output_dirs = [job.output_dir.resolve() for job in jobs]
    if len(set(job_ids)) != len(job_ids) or len(set(output_dirs)) != len(output_dirs):
        raise ValueError("Diagnostic job identifiers and output directories must be unique")
    phase0_results = (phase0_path.parents[1] / phase0["experiment"]["results_dir"] /
                      phase0["experiment"]["name"]).resolve()
    if any(path == phase0_results or phase0_results in path.parents for path in output_dirs):
        raise ValueError("A diagnostic output directory overlaps the original Phase 0 results")
    return jobs


def inspect_run(job: DiagnosticJob) -> RunStatus:
    """Classify a target as pending, validated complete, or ambiguous partial."""
    path = job.output_dir
    if not path.exists() or (path.is_dir() and not any(path.iterdir())):
        return RunStatus(RunState.PENDING, "output directory is absent or empty")
    if not path.is_dir():
        return RunStatus(RunState.PARTIAL, "output path is not a directory")
    try:
        _validate_complete_run(job)
    except (ValueError, OSError, KeyError, json.JSONDecodeError, yaml.YAMLError) as exc:
        return RunStatus(RunState.PARTIAL, str(exc))
    return RunStatus(RunState.COMPLETE, "all required artifacts validated")


def _validate_complete_run(job: DiagnosticJob) -> None:
    path = job.output_dir
    required = [*CSV_REQUIREMENTS, *NONEMPTY_ARTIFACTS, "resolved_config.yaml", "seed.txt"]
    missing = [name for name in required if not (path / name).is_file() or (path / name).stat().st_size == 0]
    if missing:
        raise ValueError(f"missing or empty artifacts: {', '.join(missing)}")

    expected_config = copy.deepcopy(job.config)
    expected_config["run"] = {
        "context_feature": job.feature,
        "observation_mode": job.mode,
        "seed": job.seed,
    }
    saved_config = _load_yaml(path / "resolved_config.yaml")
    if saved_config != expected_config:
        raise ValueError("resolved_config.yaml does not match the planned job")
    if (path / "seed.txt").read_text(encoding="ascii").strip() != str(job.seed):
        raise ValueError("seed.txt does not match the planned seed")
    with (path / "metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError("metadata.json must contain an object")
    if not isinstance(_load_yaml(path / "contexts.yaml"), dict):
        raise ValueError("contexts.yaml must contain a mapping")
    if not zipfile.is_zipfile(path / "model.zip"):
        raise ValueError("model.zip is not a valid ZIP checkpoint")
    for filename, columns in CSV_REQUIREMENTS.items():
        _validate_csv(path / filename, columns)


def _validate_csv(path: Path, required_columns: set[str]) -> None:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required_columns - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        if next(reader, None) is None:
            raise ValueError(f"{path.name} contains no data rows")


def _validate_configs(
    phase0_path: Path,
    phase0: dict[str, Any],
    configs: list[tuple[Path, dict[str, Any]]],
) -> None:
    if phase0_path.name != "phase0.yaml":
        raise ValueError("Diagnostic matrix must explicitly reference configs/phase0.yaml")
    by_name = {str(config["experiment"]["name"]): (path, config) for path, config in configs}
    if set(by_name) != set(EXPECTED_EXPERIMENTS) or len(by_name) != len(configs):
        raise ValueError(f"Diagnostic experiments must be exactly {sorted(EXPECTED_EXPERIMENTS)}")

    baseline_training = {key: value for key, value in phase0["training"].items() if key != "total_timesteps"}
    original_splits = phase0["environment"]["splits"]
    low = float(original_splits["train"]["low_multiplier"])
    high = float(original_splits["train"]["high_multiplier"])
    expected_values = {"specialist_low_300k": low, "specialist_high_300k": high}
    for name, (path, config) in by_name.items():
        kind, fixed_value, timesteps, modes = EXPECTED_EXPERIMENTS[name]
        if name in expected_values:
            fixed_value = expected_values[name]
        experiment = config["experiment"]
        if experiment["results_dir"] != "results/phase0_diagnostic":
            raise ValueError(f"{path} must use results/phase0_diagnostic")
        if experiment["seeds"] != [0, 1] or experiment["modes"] != list(modes):
            raise ValueError(f"{path} has an invalid seeds or modes matrix")
        if experiment["context_features"] != ["length"]:
            raise ValueError(f"{path} must vary only length")
        if int(config["training"]["total_timesteps"]) != timesteps:
            raise ValueError(f"{path} has the wrong total_timesteps")
        changed_training = {key: value for key, value in config["training"].items() if key != "total_timesteps"}
        if changed_training != baseline_training:
            raise ValueError(f"{path} changes a PPO hyperparameter other than total_timesteps")
        for section in ("evaluation", "reproducibility"):
            if config[section] != phase0[section]:
                raise ValueError(f"{path} changes the Phase 0 {section} settings")
        if config["environment"]["split_seed"] != phase0["environment"]["split_seed"]:
            raise ValueError(f"{path} changes the split seed")
        if config["environment"]["oracle_normalization"] != phase0["environment"]["oracle_normalization"]:
            raise ValueError(f"{path} changes oracle normalization")
        splits = config["environment"]["splits"]
        if kind == "contextual" and splits != original_splits:
            raise ValueError(f"{path} does not exactly reuse the original context splits")
        if kind == "fixed" and splits != {"train": {"values": [fixed_value]}}:
            raise ValueError(f"{path} must train and evaluate only at length={fixed_value}")


def _relative_to(parent: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (parent / path).resolve()


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value
