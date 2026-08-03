from __future__ import annotations

from pathlib import Path

import pytest

from crl_ood.utils.metadata import load_config


@pytest.fixture
def smoke_config() -> dict:
    return load_config(Path(__file__).parents[1] / "configs" / "smoke.yaml")
