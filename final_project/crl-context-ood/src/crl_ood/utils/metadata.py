"""Configuration loading and run metadata capture."""

from __future__ import annotations

import copy
import importlib.metadata
import json
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
            "python": platform.python_version(),
        },
    }


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
