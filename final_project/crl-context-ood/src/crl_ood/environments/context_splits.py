"""Deterministic context splits for the Phase 0 Pendulum experiments."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

FEATURE_KEYS = {"gravity": "g", "length": "l", "dt": "dt"}
FEATURE_DEFAULTS = {"gravity": 10.0, "length": 1.0, "dt": 0.05}
SPLIT_NAMES = ("train", "id_test", "ood_low", "ood_high")


def carl_feature_key(feature: str) -> str:
    """Translate a user-facing candidate name to CARL's Pendulum key."""
    try:
        return FEATURE_KEYS[feature]
    except KeyError as exc:
        supported = ", ".join(FEATURE_KEYS)
        raise ValueError(f"Unsupported context feature {feature!r}; use {supported}") from exc


def build_context_splits(
    feature: str,
    split_config: Mapping[str, Mapping[str, float | int]],
    seed: int,
) -> dict[str, dict[int, dict[str, float]]]:
    """Build seeded, explicit context sets while retaining CARL defaults."""
    key = carl_feature_key(feature)
    default = FEATURE_DEFAULTS[feature]
    rng = np.random.default_rng(seed)
    splits: dict[str, dict[int, dict[str, float]]] = {}

    for split_name in SPLIT_NAMES:
        if split_name not in split_config:
            raise ValueError(f"Missing environment split configuration: {split_name}")
        spec = split_config[split_name]
        count = int(spec["count"])
        if count < 1:
            raise ValueError(f"{split_name}.count must be positive")
        low = default * float(spec["low_multiplier"])
        high = default * float(spec["high_multiplier"])
        if not 0 < low <= high:
            raise ValueError(f"Invalid positive range for {split_name}: [{low}, {high}]")

        values = np.linspace(low, high, count, dtype=np.float64)
        values = values[rng.permutation(count)]
        splits[split_name] = {
            context_id: {key: float(value)}
            for context_id, value in enumerate(values)
        }

    _validate_ood_disjoint(splits, key)
    return splits


def context_values(contexts: Mapping[int, Mapping[str, float]], feature: str) -> list[float]:
    """Return context values in their deterministic assignment order."""
    key = carl_feature_key(feature)
    return [float(context[key]) for context in contexts.values()]


def _validate_ood_disjoint(
    splits: Mapping[str, Mapping[int, Mapping[str, float]]], key: str
) -> None:
    train = np.asarray([context[key] for context in splits["train"].values()])
    for split_name in ("ood_low", "ood_high"):
        ood = np.asarray([context[key] for context in splits[split_name].values()])
        if np.any(np.isclose(train[:, None], ood[None, :])):
            raise ValueError(f"Train and {split_name} context values must be disjoint")
        if split_name == "ood_low" and not float(ood.max()) < float(train.min()):
            raise ValueError("ood_low range must be strictly below the train range")
        if split_name == "ood_high" and not float(ood.min()) > float(train.max()):
            raise ValueError("ood_high range must be strictly above the train range")
