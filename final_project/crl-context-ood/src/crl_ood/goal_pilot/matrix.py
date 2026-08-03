"""Build and validate the unique target-angle pilot job matrix."""

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

from crl_ood.goal_pilot.spec import EXPECTED_SPLITS, goal_splits
from crl_ood.utils.paths import project_root

CSV_REQUIREMENTS = {
    "contexts.csv": {"run_id", "method", "split", "target_angle", "normalized_target_angle"},
    "evaluation_plan.csv": {"run_id", "method", "seed", "split", "target_angle", "episode_seed"},
    "training_metrics.csv": {"run_id", "method", "seed", "environment_steps", "episode_return"},
    "episode_returns.csv": {"run_id", "method", "seed", "split", "target_angle", "episode_seed", "return"},
    "context_returns.csv": {"run_id", "method", "seed", "split", "target_angle", "mean_return"},
}
NONEMPTY_ARTIFACTS = (
    "metadata.json",
    "contexts.yaml",
    "model.zip",
    "sb3_monitor.csv",
    "sb3_logs/progress.csv",
    "run.log",
)


@dataclass(frozen=True)
class GoalPilotJob:
    job_id: str
    config_path: Path
    config: dict[str, Any]
    kind: str
    mode: str
    seed: int
    training_goals: tuple[float, ...]
    roles: tuple[str, ...]
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


def build_goal_pilot_matrix(matrix_path: str | Path) -> list[GoalPilotJob]:
    """Return 12 unique jobs, including explicit center-hidden reuse."""
    matrix_path = Path(matrix_path).resolve()
    manifest = _load_yaml(matrix_path)
    config_path = _resolve(matrix_path.parent, manifest["pilot_config"])
    phase0_path = _resolve(matrix_path.parent, manifest["phase0_reference"])
    config = _load_yaml(config_path)
    phase0 = _load_yaml(phase0_path)
    _validate_config(config, phase0, manifest)

    jobs: list[GoalPilotJob] = []
    train = tuple(EXPECTED_SPLITS["train"])
    for seed in (0, 1):
        for mode in ("hidden", "oracle"):
            jobs.append(_job(config_path, config, "contextual", mode, seed, train, ("contextual",)))
    labels = {-0.6: "negative", 0.0: "center", 0.6: "positive"}
    for goal in (-0.6, 0.0, 0.6):
        for seed in (0, 1):
            roles = ("specialist", "fixed_center_hidden") if goal == 0.0 else ("specialist",)
            jobs.append(
                _job(config_path, config, f"specialist_{labels[goal]}", "hidden", seed, (goal,), roles)
            )
    for seed in (0, 1):
        jobs.append(
            _job(config_path, config, "fixed_center", "oracle", seed, (0.0,), ("fixed_center_oracle",))
        )

    ids = [job.job_id for job in jobs]
    outputs = [job.output_dir.resolve() for job in jobs]
    if len(jobs) != 12 or len(set(ids)) != 12 or len(set(outputs)) != 12:
        raise ValueError("Goal-pilot matrix must contain exactly 12 unique atomic runs")
    center_hidden = [job for job in jobs if "fixed_center_hidden" in job.roles]
    if len(center_hidden) != 2 or any(job.kind != "specialist_center" for job in center_hidden):
        raise ValueError("Fixed-center hidden must explicitly reuse the two center specialists")
    protected = tuple(
        (project_root() / path).resolve()
        for path in ("results/phase0", "results/phase0_diagnostic", "results/phase0_audit")
    )
    if any(any(root == path or root in path.parents for root in protected) for path in outputs):
        raise ValueError("Goal-pilot outputs overlap a protected completed result tree")
    return jobs


def inspect_run(job: GoalPilotJob) -> RunStatus:
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


def resolved_job_config(job: GoalPilotJob) -> dict[str, Any]:
    resolved = copy.deepcopy(job.config)
    resolved["run"] = {
        "job_id": job.job_id,
        "kind": job.kind,
        "observation_mode": job.mode,
        "seed": job.seed,
        "training_goals": list(job.training_goals),
        "roles": list(job.roles),
    }
    return resolved


def _job(
    config_path: Path,
    config: dict[str, Any],
    kind: str,
    mode: str,
    seed: int,
    training_goals: tuple[float, ...],
    roles: tuple[str, ...],
) -> GoalPilotJob:
    goal_token = "all_train" if len(training_goals) > 1 else _goal_token(training_goals[0])
    job_id = f"{kind}__{goal_token}__{mode}__seed_{seed}"
    root = Path(config["experiment"]["results_dir"])
    if not root.is_absolute():
        root = project_root() / root
    return GoalPilotJob(
        job_id=job_id,
        config_path=config_path,
        config=config,
        kind=kind,
        mode=mode,
        seed=seed,
        training_goals=training_goals,
        roles=roles,
        total_timesteps=int(config["training"]["total_timesteps"]),
        output_dir=root / "runs" / job_id,
    )


def _goal_token(value: float) -> str:
    sign = "neg" if value < 0 else "pos" if value > 0 else "zero"
    magnitude = f"{abs(value):.1f}".replace(".", "p")
    return f"goal_{sign}_{magnitude}"


def _validate_complete_run(job: GoalPilotJob) -> None:
    path = job.output_dir
    required = [*CSV_REQUIREMENTS, *NONEMPTY_ARTIFACTS, "resolved_config.yaml", "seed.txt"]
    missing = [name for name in required if not (path / name).is_file() or (path / name).stat().st_size == 0]
    if missing:
        raise ValueError(f"missing or empty artifacts: {', '.join(missing)}")
    if _load_yaml(path / "resolved_config.yaml") != resolved_job_config(job):
        raise ValueError("resolved_config.yaml does not match the planned job")
    if (path / "seed.txt").read_text(encoding="ascii").strip() != str(job.seed):
        raise ValueError("seed.txt does not match the planned seed")
    with (path / "metadata.json").open(encoding="utf-8") as handle:
        if not isinstance(json.load(handle), dict):
            raise ValueError("metadata.json must contain an object")
    if not isinstance(_load_yaml(path / "contexts.yaml"), dict):
        raise ValueError("contexts.yaml must contain a mapping")
    if not zipfile.is_zipfile(path / "model.zip"):
        raise ValueError("model.zip is not a valid ZIP checkpoint")
    for filename, columns in CSV_REQUIREMENTS.items():
        _validate_csv(path / filename, columns)


def _validate_csv(path: Path, columns: set[str]) -> None:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = columns - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        if next(reader, None) is None:
            raise ValueError(f"{path.name} contains no data rows")


def _validate_config(config: dict[str, Any], phase0: dict[str, Any], manifest: dict[str, Any]) -> None:
    if config["experiment"] != {
        "name": "goal_pilot",
        "results_dir": "results/goal_pilot",
        "seeds": [0, 1],
        "modes": ["hidden", "oracle"],
    }:
        raise ValueError("Pilot experiment namespace or matrix is invalid")
    goal_splits(config)
    if int(config["training"]["total_timesteps"]) != 300_000:
        raise ValueError("Every real pilot job must use 300,000 timesteps")
    ppo = {key: value for key, value in config["training"].items() if key != "total_timesteps"}
    baseline = {key: value for key, value in phase0["training"].items() if key != "total_timesteps"}
    if ppo != baseline:
        raise ValueError("Pilot PPO hyperparameters must match configs/phase0.yaml")
    for section in ("evaluation", "reproducibility"):
        if config[section] != phase0[section]:
            raise ValueError(f"Pilot {section} must match configs/phase0.yaml")
    expected_jobs = {
        "contextual": {"modes": ["hidden", "oracle"], "seeds": [0, 1], "training_goals": "all_train"},
        "specialists": {"modes": ["hidden"], "seeds": [0, 1], "training_goals": [-0.6, 0.0, 0.6]},
        "fixed_center": {"modes": ["hidden", "oracle"], "seeds": [0, 1], "training_goals": [0.0], "hidden_reuses": "specialist_center"},
    }
    if manifest.get("jobs") != expected_jobs:
        raise ValueError("matrix.yaml must retain the exact predeclared pilot matrix and deduplication")


def _resolve(parent: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (parent / path).resolve()


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value
