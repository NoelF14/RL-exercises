"""Observation modes for the CARL Pendulum environment."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class ContextObservation(gym.ObservationWrapper):
    """Expose state alone or state concatenated with one active context value."""

    def __init__(self, env: gym.Env, mode: str):
        super().__init__(env)
        if mode not in {"hidden", "oracle"}:
            raise ValueError("Observation mode must be 'hidden' or 'oracle'")
        self.mode = mode

        state_space = env.observation_space["obs"]
        if not isinstance(state_space, gym.spaces.Box):
            raise TypeError("CARLPendulum state observation must be a Box")
        low = np.asarray(state_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(state_space.high, dtype=np.float32).reshape(-1)
        if mode == "oracle":
            context_space = env.observation_space["context"]
            low = np.concatenate((low, np.asarray(context_space.low, dtype=np.float32)))
            high = np.concatenate((high, np.asarray(context_space.high, dtype=np.float32)))
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation: dict[str, Any]) -> np.ndarray:
        state = np.asarray(observation["obs"], dtype=np.float32).reshape(-1)
        if self.mode == "hidden":
            return state
        context = np.asarray(observation["context"], dtype=np.float32).reshape(-1)
        return np.concatenate((state, context), dtype=np.float32)

    @property
    def active_context(self) -> dict[str, float]:
        return dict(self.env.context)

    @property
    def active_context_id(self) -> int | None:
        return self.env.context_id
