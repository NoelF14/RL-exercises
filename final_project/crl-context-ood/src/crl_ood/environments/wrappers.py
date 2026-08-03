"""Observation modes for the CARL Pendulum environment."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class ContextObservation(gym.ObservationWrapper):
    """Expose state alone or state concatenated with one active context value."""

    def __init__(
        self,
        env: gym.Env,
        mode: str,
        context_center: float,
        context_scale: float,
    ):
        super().__init__(env)
        if mode not in {"hidden", "oracle"}:
            raise ValueError("Observation mode must be 'hidden' or 'oracle'")
        self.mode = mode
        if context_scale <= 0:
            raise ValueError("Context normalization scale must be positive")
        self.context_center = float(context_center)
        self.context_scale = float(context_scale)

        state_space = env.observation_space["obs"]
        if not isinstance(state_space, gym.spaces.Box):
            raise TypeError("CARLPendulum state observation must be a Box")
        low = np.asarray(state_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(state_space.high, dtype=np.float32).reshape(-1)
        if mode == "oracle":
            context_space = env.observation_space["context"]
            context_low = (
                np.asarray(context_space.low, dtype=np.float32) - self.context_center
            ) / self.context_scale
            context_high = (
                np.asarray(context_space.high, dtype=np.float32) - self.context_center
            ) / self.context_scale
            low = np.concatenate((low, context_low))
            high = np.concatenate((high, context_high))
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation: dict[str, Any]) -> np.ndarray:
        state = np.asarray(observation["obs"], dtype=np.float32).reshape(-1)
        if self.mode == "hidden":
            return state
        context = np.asarray(observation["context"], dtype=np.float32).reshape(-1)
        normalized = (context - self.context_center) / self.context_scale
        return np.concatenate((state, normalized), dtype=np.float32)

    @property
    def active_context(self) -> dict[str, float]:
        return dict(self.env.context)

    @property
    def active_context_id(self) -> int | None:
        return self.env.context_id

    @property
    def normalized_active_context(self) -> float:
        raw = float(self.env.context[self.env.obs_context_features[0]])
        return (raw - self.context_center) / self.context_scale
