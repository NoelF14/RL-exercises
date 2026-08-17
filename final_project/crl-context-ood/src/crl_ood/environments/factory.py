"""Factory for the Phase 0 CARLPendulum environment."""

from __future__ import annotations

from collections.abc import Mapping

import gymnasium as gym
from carl.context.selection import StaticSelector
from carl.envs import CARLPendulum, CARLCartPole

from crl_ood.environments.context_splits import carl_feature_key
from crl_ood.environments.wrappers import ContextObservation

def make_cartpole_env(
    contexts: Mapping[int, Mapping[str, float]],
    feature: str,
    mode: str,
    seed: int,
    *,
    context_normalization: tuple[float, float],
    static_context: bool = False,
) -> ContextObservation:
    """Create a seeded CARLCartPole with a Phase 0 observation mode."""
    if not contexts:
        raise ValueError("At least one context is required")
    context_dict = complete_carl_cartpole_contexts(contexts)
    base_env = gym.make("CartPole-v1", render_mode="rgb_array")
    carl_env = CARLCartPole(
        env=base_env,
        contexts=context_dict,
        obs_context_features=[carl_feature_key(feature)],
        obs_context_as_dict=False,
        context_selector=StaticSelector if static_context else None,
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
    """Create a seeded CARLPendulum with a Phase 0 observation mode."""
    if not contexts:
        raise ValueError("At least one context is required")
    context_dict = complete_carl_pendulum_contexts(contexts)
    base_env = gym.make("Pendulum-v1", render_mode="rgb_array")
    carl_env = CARLPendulum(
        env=base_env,
        contexts=context_dict,
        obs_context_features=[carl_feature_key(feature)],
        obs_context_as_dict=False,
        context_selector=StaticSelector if static_context else None,
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
    """Expand partial contexts exactly as CARL's context setter does."""
    defaults = CARLPendulum.get_default_context()
    completed: dict[int, dict[str, float]] = {}
    for context_id, context in contexts.items():
        full_context = {str(key): float(value) for key, value in defaults.items()}
        full_context.update({str(key): float(value) for key, value in context.items()})
        completed[int(context_id)] = full_context
    return completed

def complete_carl_cartpole_contexts(
    contexts: Mapping[int, Mapping[str, float]],
) -> dict[int, dict[str, float]]:
    """Expand partial contexts exactly as CARL's context setter does."""
    defaults = CARLCartPole.get_default_context()
    completed: dict[int, dict[str, float]] = {}
    for context_id, context in contexts.items():
        full_context = {str(key): float(value) for key, value in defaults.items()}
        full_context.update({str(key): float(value) for key, value in context.items()})
        completed[int(context_id)] = full_context
    return completed
