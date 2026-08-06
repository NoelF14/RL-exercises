"""Pure helpers for the immutable PointRobot context and gate specification."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import math

SPLIT_ORDER = ("train", "id", "ood_left", "ood_right")
PRIMARY_SPLITS = ("train", "id")
OOD_SPLITS = ("ood_left", "ood_right")
EXPECTED_SPLITS = {
    "train": (-0.6, -0.3, 0.0, 0.3, 0.6),
    "id": (-0.45, -0.15, 0.15, 0.45),
    "ood_left": (-1.0, -0.8),
    "ood_right": (0.8, 1.0),
}


def context_splits(config: Mapping[str, Any]) -> dict[str, dict[int, float]]:
    raw = config["environment"]["splits"]
    splits = {name: {index: float(value) for index, value in enumerate(raw[name])} for name in SPLIT_ORDER}
    actual = {name: tuple(values.values()) for name, values in splits.items()}
    if actual != EXPECTED_SPLITS:
        raise ValueError(f"PointRobot splits must exactly equal {EXPECTED_SPLITS}")
    flat = [value for name in SPLIT_ORDER for value in actual[name]]
    if len(flat) != len(set(flat)):
        raise ValueError("PointRobot context splits must be pairwise disjoint")
    if not all(min(actual["train"]) < x < max(actual["train"]) for x in actual["id"]):
        raise ValueError("ID goals must lie inside the training range")
    if not max(actual["ood_left"]) < min(actual["train"]):
        raise ValueError("OOD-left must stay below the training range")
    if not min(actual["ood_right"]) > max(actual["train"]):
        raise ValueError("OOD-right must stay above the training range")
    return splits


def context_record(split: str, context_id: int, angle: float) -> dict[str, Any]:
    return {
        "split": split,
        "context_id": context_id,
        "goal_angle": float(angle),
        "goal_cos": math.cos(angle),
        "goal_sin": math.sin(angle),
        "goal_x": math.cos(angle),
        "goal_y": math.sin(angle),
    }


def circular_distance(predicted: float, target: float) -> float:
    """Absolute shortest angular distance in radians."""
    return abs((float(predicted) - float(target) + math.pi) % (2.0 * math.pi) - math.pi)

