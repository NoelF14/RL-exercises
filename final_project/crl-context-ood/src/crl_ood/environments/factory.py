"""Factories for supported CARL environments."""

from __future__ import annotations

from collections.abc import Mapping

import gymnasium as gym
from carl.context.selection import StaticSelector
from carl.envs import CARLCartPole, CARLPendulum

from crl_ood.environments.context_splits import (
    carl_feature_key,
    infer_environment_from_contexts,
    normalize_environment_name,
)
from crl_ood.environments.wrappers import ContextObservation


def make_env(
    environment: str,
    contexts: Mapping[int, Mapping[str, float]],
    feature: str,
    mode: str,
    seed: int,
    *,
    context_normalization: tuple[float, float],
    static_context: bool = False,
) -> ContextObservation:
    """Create one supported CARL environment."""
    environment = normalize_environment_name(environment)

    if environment == "pendulum":
        return make_pendulum_env(
            contexts,
            feature,
            mode,
            seed,
            context_normalization=context_normalization,
            static_context=static_context,
        )

    if environment == "cartpole":
        return make_cartpole_env(
            contexts,
            feature,
            mode,
            seed,
            context_normalization=context_normalization,
            static_context=static_context,
        )

    raise AssertionError(f"Unhandled environment {environment!r}")


def make_cartpole_env(
    contexts: Mapping[int, Mapping[str, float]],
    feature: str,
    mode: str,
    seed: int,
    *,
    context_normalization: tuple[float, float],
    static_context: bool = False,
) -> ContextObservation:
    """Create a seeded CARLCartPole."""
    if not contexts:
        raise ValueError("At least one context is required")

    context_dict = complete_carl_cartpole_contexts(contexts)

    base_env = gym.make(
        "CartPole-v1",
        render_mode="rgb_array",
    )

    carl_env = CARLCartPole(
        env=base_env,
        contexts=context_dict,
        obs_context_features=[
            carl_feature_key(feature, "cartpole")
        ],
        obs_context_as_dict=False,
        context_selector=(
            StaticSelector
            if static_context
            else None
        ),
    )

    env = ContextObservation(
        carl_env,
        mode=mode,
        context_center=context_normalization[0],
        context_scale=context_normalization[1],
    )

    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def make_pendulum_env(
    contexts: Mapping[int, Mapping[str, float]],
    feature: str,
    mode: str,
    seed: int,
    *,
    context_normalization: tuple[float, float],
    static_context: bool = False,
) -> ContextObservation:
    """Create a seeded CARLPendulum."""
    if not contexts:
        raise ValueError("At least one context is required")

    context_dict = complete_carl_pendulum_contexts(contexts)

    base_env = gym.make(
        "Pendulum-v1",
        render_mode="rgb_array",
    )

    carl_env = CARLPendulum(
        env=base_env,
        contexts=context_dict,
        obs_context_features=[
            carl_feature_key(feature, "pendulum")
        ],
        obs_context_as_dict=False,
        context_selector=(
            StaticSelector
            if static_context
            else None
        ),
    )

    env = ContextObservation(
        carl_env,
        mode=mode,
        context_center=context_normalization[0],
        context_scale=context_normalization[1],
    )

    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def complete_carl_pendulum_contexts(
    contexts: Mapping[int, Mapping[str, float]],
) -> dict[int, dict[str, float]]:
    """Expand partial contexts using CARLPendulum defaults."""
    defaults = CARLPendulum.get_default_context()
    completed: dict[int, dict[str, float]] = {}

    for context_id, context in contexts.items():
        full_context = {
            str(key): float(value)
            for key, value in defaults.items()
        }
        full_context.update(
            {
                str(key): float(value)
                for key, value in context.items()
            }
        )
        completed[int(context_id)] = full_context

    return completed


def complete_carl_cartpole_contexts(
    contexts: Mapping[int, Mapping[str, float]],
) -> dict[int, dict[str, float]]:
    """Expand partial contexts using CARLCartPole defaults."""
    defaults = CARLCartPole.get_default_context()
    completed: dict[int, dict[str, float]] = {}

    for context_id, context in contexts.items():
        full_context = {
            str(key): float(value)
            for key, value in defaults.items()
        }
        full_context.update(
            {
                str(key): float(value)
                for key, value in context.items()
            }
        )
        completed[int(context_id)] = full_context

    return completed


def complete_carl_contexts(
    contexts: Mapping[int, Mapping[str, float]],
) -> dict[int, dict[str, float]]:
    """Expand contexts using the environment implied by their feature keys."""
    environment = infer_environment_from_contexts(contexts)

    if environment == "cartpole":
        return complete_carl_cartpole_contexts(contexts)

    return complete_carl_pendulum_contexts(contexts)
