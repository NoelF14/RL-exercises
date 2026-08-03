"""Build and validate the two fixed-center oracle sanity jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from crl_ood.diagnostic.matrix import DiagnosticJob, RunStatus, inspect_run
from crl_ood.utils.paths import project_root, run_directory


def build_fixed_center_matrix(matrix_path: str | Path) -> list[DiagnosticJob]:
    """Return the unique seed-0/seed-1 fixed-center oracle jobs."""
    matrix_path = Path(matrix_path).resolve()
    manifest = _load_yaml(matrix_path)
    phase0_path = _resolve(matrix_path.parent, manifest["phase0_config"])
    default_path = _resolve(matrix_path.parent, manifest["default_300k_config"])
    config_path = _resolve(matrix_path.parent, manifest["config"])
    phase0 = _load_yaml(phase0_path)
    default = _load_yaml(default_path)
    config = _load_yaml(config_path)
    _validate_config(phase0, default, config)

    jobs = []
    for seed in config["experiment"]["seeds"]:
        seed = int(seed)
        jobs.append(
            DiagnosticJob(
                job_id=f"fixed_center_oracle_300k__length__oracle__seed_{seed}",
                config_path=config_path,
                config=config,
                experiment="fixed_center_oracle_300k",
                feature="length",
                mode="oracle",
                seed=seed,
                total_timesteps=300_000,
                output_dir=run_directory(config, "length", "oracle", seed),
            )
        )
    outputs = [job.output_dir.resolve() for job in jobs]
    if len(jobs) != 2 or len(set(outputs)) != 2:
        raise ValueError("Fixed-center audit must define exactly two unique jobs")
    protected = {
        (project_root() / phase0["experiment"]["results_dir"] / phase0["experiment"]["name"]).resolve(),
        (project_root() / "results/phase0_diagnostic").resolve(),
    }
    if any(any(root == path or root in path.parents for root in protected) for path in outputs):
        raise ValueError("Fixed-center outputs overlap a completed results namespace")
    return jobs


def inspect_fixed_center_run(job: DiagnosticJob) -> RunStatus:
    """Classify one job using the same artifact validation as diagnostic runs."""
    return inspect_run(job)


def _validate_config(
    phase0: dict[str, Any], default: dict[str, Any], config: dict[str, Any]
) -> None:
    experiment = config["experiment"]
    if experiment != {
        "name": "fixed_center_oracle_300k",
        "results_dir": "results/phase0_audit",
        "seeds": [0, 1],
        "modes": ["oracle"],
        "context_features": ["length"],
    }:
        raise ValueError("Fixed-center experiment matrix is not the required two-job oracle matrix")
    if config["training"] != default["training"]:
        raise ValueError("Fixed-center PPO settings must exactly match default_300k")
    for section in ("evaluation", "reproducibility"):
        if config[section] != default[section] or config[section] != phase0[section]:
            raise ValueError(f"Fixed-center {section} must match Phase 0 and default_300k")
    environment = config["environment"]
    if environment != {
        "split_seed": phase0["environment"]["split_seed"],
        "oracle_normalization": "train_range",
        "splits": {"train": {"values": [1.0]}},
    }:
        raise ValueError("Fixed-center environment must be CARLPendulum length=1.0 only")


def _resolve(parent: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (parent / path).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value
