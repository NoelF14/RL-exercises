"""Project and run path helpers."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def run_directory(config: dict[str, Any], feature: str, mode: str, seed: int) -> Path:
    """Return an atomic run path without creating or modifying it."""
    configured = Path(config["experiment"]["results_dir"])
    results_root = configured if configured.is_absolute() else project_root() / configured
    return results_root / config["experiment"]["name"] / feature / mode / f"seed_{seed}"


def prepare_run_directory(
    config: dict[str, Any],
    feature: str,
    mode: str,
    seed: int,
    *,
    overwrite: bool = False,
) -> Path:
    """Create an empty atomic run directory, refusing stale artifact mixing."""
    path = run_directory(config, feature, mode, seed)
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Run directory already contains artifacts: {path}. "
                "Pass --overwrite to replace this atomic run."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def assert_run_available(
    config: dict[str, Any], feature: str, mode: str, seed: int
) -> None:
    path = run_directory(config, feature, mode, seed)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Run directory already contains artifacts: {path}. "
            "Pass --overwrite to replace this atomic run."
        )


def run_identifier(config: dict[str, Any], feature: str, method: str, seed: int) -> str:
    return f"{config['experiment']['name']}__{feature}__{method}__seed_{seed}"
