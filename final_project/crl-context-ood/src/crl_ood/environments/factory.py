"""Factory for the Phase 0 CARLPendulum environment."""

from __future__ import annotations

from collections.abc import Mapping

import gymnasium as gym
from carl.context.selection import StaticSelector
from carl.envs import CARLPendulum

from crl_ood.environments.context_splits import carl_feature_key
from crl_ood.environments.wrappers import ContextObservation


def make_pendulum_env(
    contexts: Mapping[int, Mapping[str, float]],
    feature: str,
    mode: str,
    seed: int,
    *,
    static_context: bool = False,
) -> ContextObservation:
    """Create a seeded CARLPendulum with a Phase 0 observation mode."""
    if not contexts:
        raise ValueError("At least one context is required")
    context_dict = {
        int(context_id): {str(key): float(value) for key, value in context.items()}
        for context_id, context in contexts.items()
    }
    base_env = gym.make("Pendulum-v1", render_mode="rgb_array")
    carl_env = CARLPendulum(
        env=base_env,
        contexts=context_dict,
        obs_context_features=[carl_feature_key(feature)],
        obs_context_as_dict=False,
        context_selector=StaticSelector if static_context else None,
    )
    env = ContextObservation(carl_env, mode=mode)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env
