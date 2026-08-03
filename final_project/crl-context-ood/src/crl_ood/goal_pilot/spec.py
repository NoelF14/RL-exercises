"""Pure configuration helpers for the predeclared goal pilot."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

SPLIT_ORDER = ("train", "id_test", "ood_left", "ood_right")
EXPECTED_SPLITS = {
    "train": (-0.6, -0.3, 0.0, 0.3, 0.6),
    "id_test": (-0.45, -0.15, 0.15, 0.45),
    "ood_left": (-1.0, -0.8),
    "ood_right": (0.8, 1.0),
}


def goal_splits(config: Mapping[str, Any]) -> dict[str, dict[int, float]]:
    raw = config["environment"]["splits"]
    splits = {
        name: {index: float(value) for index, value in enumerate(raw[name])}
        for name in SPLIT_ORDER
    }
    actual = {name: tuple(values.values()) for name, values in splits.items()}
    if actual != EXPECTED_SPLITS:
        raise ValueError(f"Goal splits must exactly equal the predeclared values: {EXPECTED_SPLITS}")
    seen: list[float] = []
    for name in SPLIT_ORDER:
        for value in actual[name]:
            if any(abs(value - previous) < 1e-12 for previous in seen):
                raise ValueError("Target-angle splits must be pairwise disjoint")
            seen.append(value)
    train = actual["train"]
    if not all(min(train) < value < max(train) for value in actual["id_test"]):
        raise ValueError("ID goals must lie strictly inside the training extrema")
    if not max(actual["ood_left"]) < min(train):
        raise ValueError("OOD-left goals must lie below the training range")
    if not min(actual["ood_right"]) > max(train):
        raise ValueError("OOD-right goals must lie above the training range")
    return splits


def goal_normalization(train_goals: Mapping[int, float] | Sequence[float]) -> tuple[float, float]:
    values = list(train_goals.values()) if isinstance(train_goals, Mapping) else list(train_goals)
    if not values:
        raise ValueError("Training goals cannot be empty")
    low, high = min(map(float, values)), max(map(float, values))
    if high <= low:
        raise ValueError("Normalization requires distinct minimum and maximum training goals")
    return (low + high) / 2.0, (high - low) / 2.0
