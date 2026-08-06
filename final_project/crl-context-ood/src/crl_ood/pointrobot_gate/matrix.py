"""Build and validate the 12 unique PointRobot gate jobs."""

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

from crl_ood.pointrobot_gate.spec import EXPECTED_SPLITS, context_splits
from crl_ood.utils.paths import project_root

CSV_REQUIREMENTS = {
    "contexts.csv": {"split", "goal_angle", "goal_cos", "goal_sin", "goal_x", "goal_y"},
    "evaluation_plan.csv": {"run_id", "seed", "split", "goal_angle", "episode_seed"},
    "training_metrics.csv": {"run_id", "seed", "environment_steps", "episode_return"},
    "episode_returns.csv": {"run_id", "split", "goal_angle", "return", "success", "final_distance", "minimum_distance"},
    "context_returns.csv": {"run_id", "split", "goal_angle", "mean_return", "success_rate", "mean_final_distance"},
    "success_metrics.csv": {"run_id", "split", "goal_angle", "success_rate"},
    "distance_metrics.csv": {"run_id", "split", "goal_angle", "mean_final_distance", "mean_minimum_distance"},
}
NONEMPTY = (
    "metadata.json", "source_provenance.json", "environment_spec.yaml", "contexts.yaml",
    "model.zip", "sb3_monitor.csv", "sb3_logs/progress.csv", "run.log",
)


@dataclass(frozen=True)
class PointRobotJob:
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


def build_matrix(matrix_path: str | Path) -> list[PointRobotJob]:
    matrix_path = Path(matrix_path).resolve()
    manifest = _yaml(matrix_path)
    config_path = (matrix_path.parent / manifest["gate_config"]).resolve()
    config = _yaml(config_path)
    _validate_config(config, manifest)
    train = EXPECTED_SPLITS["train"]
    jobs: list[PointRobotJob] = []
    for seed in (0, 1):
        for mode in ("hidden", "oracle"):
            jobs.append(_job(config_path, config, "contextual", mode, seed, train, ("contextual",)))
    names = {-0.6: "negative", 0.0: "center", 0.6: "positive"}
    for goal in (-0.6, 0.0, 0.6):
        for seed in (0, 1):
            roles = ("specialist", "fixed_center_hidden") if goal == 0.0 else ("specialist",)
            jobs.append(_job(config_path, config, f"specialist_{names[goal]}", "hidden", seed, (goal,), roles))
    for seed in (0, 1):
        jobs.append(_job(config_path, config, "fixed_center", "oracle", seed, (0.0,), ("fixed_center_oracle",)))
    if len(jobs) != 12 or len({x.job_id for x in jobs}) != 12 or len({x.output_dir for x in jobs}) != 12:
        raise ValueError("PointRobot gate matrix must contain exactly 12 unique atomic runs")
    reused = [x for x in jobs if "fixed_center_hidden" in x.roles]
    if len(reused) != 2 or any(x.kind != "specialist_center" for x in reused):
        raise ValueError("Fixed-center hidden must reuse the two center specialists")
    protected = tuple((project_root() / p).resolve() for p in (
        "results/phase0", "results/phase0_diagnostic", "results/phase0_audit",
        "results/goal_pilot", "results/goal_pilot_mechanistic_audit",
    ))
    for job in jobs:
        if any(root == job.output_dir.resolve() or root in job.output_dir.resolve().parents for root in protected):
            raise ValueError("PointRobot output overlaps a protected result namespace")
    return jobs


def resolved_config(job: PointRobotJob) -> dict[str, Any]:
    value = copy.deepcopy(job.config)
    value["run"] = {
        "job_id": job.job_id, "kind": job.kind, "observation_mode": job.mode,
        "seed": job.seed, "training_goals": list(job.training_goals), "roles": list(job.roles),
    }
    return value


def inspect_run(job: PointRobotJob) -> RunStatus:
    path = job.output_dir
    if not path.exists() or (path.is_dir() and not any(path.iterdir())):
        return RunStatus(RunState.PENDING, "output directory is absent or empty")
    if not path.is_dir():
        return RunStatus(RunState.PARTIAL, "output path is not a directory")
    try:
        _validate_complete(job)
    except (ValueError, OSError, KeyError, json.JSONDecodeError, yaml.YAMLError) as exc:
        return RunStatus(RunState.PARTIAL, str(exc))
    return RunStatus(RunState.COMPLETE, "all required artifacts validated")


def _job(config_path: Path, config: dict[str, Any], kind: str, mode: str, seed: int,
         goals: tuple[float, ...], roles: tuple[str, ...]) -> PointRobotJob:
    goal = "all_train" if len(goals) > 1 else _token(goals[0])
    job_id = f"{kind}__{goal}__{mode}__seed_{seed}"
    root = Path(config["experiment"]["results_dir"])
    if not root.is_absolute():
        root = project_root() / root
    return PointRobotJob(job_id, config_path, config, kind, mode, seed, goals, roles,
                         int(config["training"]["total_timesteps"]), root / "runs" / job_id)


def _token(value: float) -> str:
    sign = "neg" if value < 0 else "pos" if value > 0 else "zero"
    return f"goal_{sign}_{abs(value):.1f}".replace(".", "p")


def _validate_complete(job: PointRobotJob) -> None:
    required = [*CSV_REQUIREMENTS, *NONEMPTY, "resolved_config.yaml", "seed.txt"]
    missing = [name for name in required if not (job.output_dir / name).is_file() or (job.output_dir / name).stat().st_size == 0]
    if missing:
        raise ValueError(f"missing or empty artifacts: {', '.join(missing)}")
    if _yaml(job.output_dir / "resolved_config.yaml") != resolved_config(job):
        raise ValueError("resolved_config.yaml does not match planned job")
    if (job.output_dir / "seed.txt").read_text(encoding="ascii").strip() != str(job.seed):
        raise ValueError("seed.txt does not match planned seed")
    for name in ("metadata.json", "source_provenance.json"):
        with (job.output_dir / name).open(encoding="utf-8") as handle:
            if not isinstance(json.load(handle), dict):
                raise ValueError(f"{name} must contain an object")
    if not zipfile.is_zipfile(job.output_dir / "model.zip"):
        raise ValueError("model.zip is not a valid ZIP checkpoint")
    for name, columns in CSV_REQUIREMENTS.items():
        with (job.output_dir / name).open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if columns - set(reader.fieldnames or ()):
                raise ValueError(f"{name} is missing required columns")
            if next(reader, None) is None:
                raise ValueError(f"{name} contains no rows")


def _validate_config(config: dict[str, Any], manifest: dict[str, Any]) -> None:
    if config["experiment"] != {"name": "pointrobot_gate", "results_dir": "results/pointrobot_gate", "seeds": [0, 1], "modes": ["hidden", "oracle"]}:
        raise ValueError("Invalid immutable PointRobot experiment namespace")
    context_splits(config)
    environment = {key: value for key, value in config["environment"].items() if key != "splits"}
    if environment != {
        "name": "DenseSemiCirclePointRobot-v0", "goal_radius": 1.0,
        "start_position": [0.0, 0.0], "reset_noise": 0.0, "step_scale": 0.1,
        "position_limit": 1.5, "horizon": 50, "action_penalty": 0.01,
        "success_threshold": 0.10, "early_termination": False,
        "reward_timing": "post_transition",
    }:
        raise ValueError("PointRobot environment defaults must remain exactly predeclared")
    if int(config["training"]["total_timesteps"]) != 200_000:
        raise ValueError("Real PointRobot gate jobs must use 200,000 PPO timesteps")
    expected_gate = {
        "primary_splits": ["train", "id"], "ood_role": "descriptive_only",
        "confidence_intervals": False, "specialist_min_own_success_rate": 0.80,
        "specialist_max_own_mean_final_distance": 0.10,
        "specialist_nearest_majority_fraction": 0.50,
        "contextual_min_oracle_success_gain": 0.20,
        "fixed_center_max_abs_success_gap": 0.10,
        "probe_history_candidates": [1, 3, 5, 10],
        "probe_long_history_candidates": [5, 10],
        "probe_min_relative_id_mae_reduction": 0.50,
        "probe_min_h1_relative_improvement": 0.10,
    }
    if config.get("gate") != expected_gate:
        raise ValueError("PointRobot gate thresholds must remain exactly predeclared")
    if config.get("probe", {}).get("history_lengths") != [1, 3, 5, 10]:
        raise ValueError("Probe history lengths must remain [1, 3, 5, 10]")
    expected = {
        "contextual": {"modes": ["hidden", "oracle"], "seeds": [0, 1], "training_goals": "all_train"},
        "specialists": {"modes": ["hidden"], "seeds": [0, 1], "training_goals": [-0.6, 0.0, 0.6]},
        "fixed_center": {"modes": ["hidden", "oracle"], "seeds": [0, 1], "training_goals": [0.0], "hidden_reuses": "specialist_center"},
    }
    if manifest.get("jobs") != expected:
        raise ValueError("matrix.yaml does not match the predeclared 12-job matrix")


def _yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value
