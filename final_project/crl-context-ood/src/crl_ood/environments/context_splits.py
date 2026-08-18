"""Deterministic context splits for CARL environments."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


FEATURE_KEYS = {
    "pendulum": {
        "gravity": "g",
        "length": "l",
        "dt": "dt",
    },
    "cartpole": {
        "length": "length",
    },
}

FEATURE_DEFAULTS = {
    "pendulum": {
        "gravity": 10.0,
        "length": 1.0,
        "dt": 0.05,
    },
    "cartpole": {
        "length": 1.0,
    },
}

SPLIT_NAMES = ("train", "id_test", "ood_low", "ood_high")


def normalize_environment_name(environment: str) -> str:
    name = str(environment).strip().lower()
    aliases = {
        "pendulum": "pendulum",
        "carlpendulum": "pendulum",
        "cartpole": "cartpole",
        "carlcartpole": "cartpole",
    }
    try:
        return aliases[name]
    except KeyError as exc:
        supported = ", ".join(sorted({"pendulum", "cartpole"}))
        raise ValueError(
            f"Unsupported environment {environment!r}; use {supported}"
        ) from exc


def carl_feature_key(feature: str, environment: str = "pendulum") -> str:
    """Translate a user-facing feature name to the selected CARL environment key."""
    environment = normalize_environment_name(environment)
    try:
        return FEATURE_KEYS[environment][feature]
    except KeyError as exc:
        supported = ", ".join(FEATURE_KEYS[environment])
        raise ValueError(
            f"Unsupported {environment} context feature {feature!r}; "
            f"use {supported}"
        ) from exc


def build_context_splits(
    feature: str,
    split_config: Mapping[str, Mapping[str, float | int]],
    seed: int,
    *,
    environment: str = "pendulum",
) -> dict[str, dict[int, dict[str, float]]]:
    """Build seeded explicit context sets while retaining environment defaults."""
    environment = normalize_environment_name(environment)
    key = carl_feature_key(feature, environment)
    default = FEATURE_DEFAULTS[environment][feature]
    rng = np.random.default_rng(seed)
    splits: dict[str, dict[int, dict[str, float]]] = {}

    explicit = all("values" in spec for spec in split_config.values())
    if explicit:
        for split_name, spec in split_config.items():
            values = np.asarray(spec["values"], dtype=np.float64)
            if values.ndim != 1 or len(values) < 1 or np.any(values <= 0):
                raise ValueError(
                    f"{split_name}.values must be a non-empty positive list"
                )
            values = values[rng.permutation(len(values))]
            splits[split_name] = {
                context_id: {key: float(value)}
                for context_id, value in enumerate(values)
            }
        return splits

    for split_name in SPLIT_NAMES:
        if split_name not in split_config:
            raise ValueError(
                f"Missing environment split configuration: {split_name}"
            )

        spec = split_config[split_name]
        count = int(spec["count"])
        if count < 1:
            raise ValueError(f"{split_name}.count must be positive")

        low = default * float(spec["low_multiplier"])
        high = default * float(spec["high_multiplier"])

        if not 0 < low <= high:
            raise ValueError(
                f"Invalid positive range for {split_name}: [{low}, {high}]"
            )

        values = np.linspace(low, high, count, dtype=np.float64)
        values = values[rng.permutation(count)]

        splits[split_name] = {
            context_id: {key: float(value)}
            for context_id, value in enumerate(values)
        }

    _validate_split_relationships(splits, key)
    return splits


def infer_environment_from_contexts(
    contexts: Mapping[int, Mapping[str, float]],
) -> str:
    """Infer CARL environment from already-expanded partial context keys."""
    if not contexts:
        raise ValueError("At least one context is required")

    keys = set(next(iter(contexts.values())).keys())

    if "length" in keys:
        return "cartpole"

    if keys & {"g", "l", "dt"}:
        return "pendulum"

    raise ValueError(
        f"Cannot infer CARL environment from context keys {sorted(keys)}"
    )


def context_values(
    contexts: Mapping[int, Mapping[str, float]],
    feature: str,
    *,
    environment: str | None = None,
) -> list[float]:
    """Return context values in deterministic assignment order."""
    if environment is None:
        environment = infer_environment_from_contexts(contexts)

    key = carl_feature_key(feature, environment)
    return [float(context[key]) for context in contexts.values()]


def context_normalization(
    train_contexts: Mapping[int, Mapping[str, float]],
    feature: str,
    *,
    environment: str | None = None,
) -> tuple[float, float]:
    """Return center and scale mapping training extrema to -1 and 1."""
    values = np.asarray(
        context_values(
            train_contexts,
            feature,
            environment=environment,
        ),
        dtype=np.float64,
    )

    center = float((values.min() + values.max()) / 2.0)
    scale = float((values.max() - values.min()) / 2.0)

    if scale == 0.0:
        scale = abs(center) if center != 0.0 else 1.0

    return center, scale


def _validate_split_relationships(
    splits: Mapping[str, Mapping[int, Mapping[str, float]]],
    key: str,
) -> None:
    train = np.asarray(
        [context[key] for context in splits["train"].values()]
    )
    identity = np.asarray(
        [context[key] for context in splits["id_test"].values()]
    )

    if (
        float(identity.min()) < float(train.min())
        or float(identity.max()) > float(train.max())
    ):
        raise ValueError("id_test values must remain inside the train range")

    all_values = {
        split_name: np.asarray(
            [context[key] for context in contexts.values()]
        )
        for split_name, contexts in splits.items()
    }

    split_names = list(all_values)

    for index, left_name in enumerate(split_names):
        for right_name in split_names[index + 1 :]:
            left = all_values[left_name]
            right = all_values[right_name]

            if np.any(
                np.isclose(
                    left[:, None],
                    right[None, :],
                )
            ):
                raise ValueError(
                    f"{left_name} and {right_name} context values "
                    "must be disjoint"
                )

    for split_name in ("ood_low", "ood_high"):
        ood = np.asarray(
            [context[key] for context in splits[split_name].values()]
        )

        if (
            split_name == "ood_low"
            and not float(ood.max()) < float(train.min())
        ):
            raise ValueError(
                "ood_low range must be strictly below the train range"
            )

        if (
            split_name == "ood_high"
            and not float(ood.min()) > float(train.max())
        ):
            raise ValueError(
                "ood_high range must be strictly above the train range"
            )
