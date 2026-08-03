"""Project and run path helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def run_directory(config: dict[str, Any], feature: str, mode: str, seed: int) -> Path:
    configured = Path(config["experiment"]["results_dir"])
    results_root = configured if configured.is_absolute() else project_root() / configured
    path = results_root / config["experiment"]["name"] / feature / mode / f"seed_{seed}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_identifier(config: dict[str, Any], feature: str, method: str, seed: int) -> str:
    return f"{config['experiment']['name']}__{feature}__{method}__seed_{seed}"
