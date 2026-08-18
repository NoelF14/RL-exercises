"""Configuration loading and run metadata capture."""

from __future__ import annotations

import copy
import csv
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import torch
import yaml

from crl_ood.utils.paths import project_root

PACKAGES = (
    "carl-bench",
    "stable-baselines3",
    "gymnasium",
    "torch",
    "numpy",
    "pandas",
    "PyYAML",
)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    return config


def resolved_run_config(
    config: dict[str, Any], feature: str, mode: str, seed: int
) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    resolved["run"] = {"context_feature": feature, "observation_mode": mode, "seed": seed}
    return resolved


def write_run_provenance(run_dir: Path, config: dict[str, Any], seed: int) -> None:
    with (run_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    (run_dir / "seed.txt").write_text(f"{seed}\n", encoding="ascii")
    with (run_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(collect_metadata(), handle, indent=2, sort_keys=True)
        handle.write("\n")


def collect_metadata() -> dict[str, Any]:
    return {
        "git": _git_metadata(),
        "packages": {
            package: _package_version(package)
            for package in PACKAGES
        },
        "device": {
            "requested_cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            "gpu_names": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
    }


def write_context_artifacts(
    run_dir: Path,
    run_id: str,
    method: str,
    feature: str,
    splits: dict[str, dict[int, dict[str, float]]],
    normalization: tuple[float, float],
    evaluation_plan: list[dict[str, Any]],
) -> None:
    """Persist exact expanded CARL contexts and the ordered evaluation plan."""
    from crl_ood.environments.context_splits import (
        carl_feature_key,
        infer_environment_from_contexts,
    )
    from crl_ood.environments.factory import complete_carl_contexts

    environment_name = infer_environment_from_contexts(splits["train"])
    key = carl_feature_key(feature, environment_name)
    manifest: dict[str, Any] = {
        "context_feature": feature,
        "carl_feature_key": key,
        "normalization": {"center": normalization[0], "scale": normalization[1]},
        "splits": {},
    }
    context_rows = []
    for split_name, contexts in splits.items():
        completed = complete_carl_contexts(contexts)
        entries = []
        for order, (context_id, context) in enumerate(completed.items()):
            entries.append(
                {
                    "order": order,
                    "context_id": context_id,
                    "context_value": context[key],
                    "carl_context": context,
                }
            )
            context_rows.append(
                {
                    "run_id": run_id,
                    "method": method,
                    "context_feature": feature,
                    "split": split_name,
                    "context_order": order,
                    "context_id": context_id,
                    "context_value": context[key],
                }
            )
        manifest["splits"][split_name] = entries

    with (run_dir / "contexts.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False)
    _write_dict_csv(
        run_dir / "contexts.csv",
        (
            "run_id",
            "method",
            "context_feature",
            "split",
            "context_order",
            "context_id",
            "context_value",
        ),
        context_rows,
    )
    _write_dict_csv(
        run_dir / "evaluation_plan.csv",
        (
            "run_id",
            "method",
            "seed",
            "context_feature",
            "context_value",
            "split",
            "context_id",
            "episode_index",
            "episode_seed",
        ),
        evaluation_plan,
    )


def load_context_manifest(
    path: str | Path,
) -> tuple[dict[str, dict[int, dict[str, float]]], tuple[float, float], str]:
    """Load the exact contexts and normalization saved before training."""
    with Path(path).open(encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    splits = {
        split_name: {
            int(entry["context_id"]): {
                str(key): float(value)
                for key, value in entry["carl_context"].items()
            }
            for entry in entries
        }
        for split_name, entries in manifest["splits"].items()
    }
    normalization = manifest["normalization"]
    return (
        splits,
        (float(normalization["center"]), float(normalization["scale"])),
        str(manifest["context_feature"]),
    )


def _write_dict_csv(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: list[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _git_metadata() -> dict[str, str | bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root(),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_root(),
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
