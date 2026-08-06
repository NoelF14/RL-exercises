"""CARLPendulum wrapper whose context changes only the reward target."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np
from carl.context.selection import StaticSelector
from carl.envs import CARLPendulum


def angle_normalize(angle: float) -> float:
    """Wrap an angle to the standard Pendulum interval [-pi, pi)."""
    return float(((angle + np.pi) % (2.0 * np.pi)) - np.pi)


def pendulum_state(observation: Any) -> np.ndarray:
    """Extract [cos(theta), sin(theta), theta_dot] from a CARL observation."""
    value = observation["obs"] if isinstance(observation, Mapping) else observation
    state = np.asarray(value, dtype=np.float32).reshape(-1)
    if state.shape != (3,):
        raise ValueError(f"Expected a three-value Pendulum state, got {state.shape}")
    return state


def target_reward(observation: Any, action: Any, target_angle: float) -> float:
    """Compute target-angle Pendulum reward from the pre-transition state."""
    state = pendulum_state(observation)
    theta = float(np.arctan2(float(state[1]), float(state[0])))
    theta_dot = float(state[2])
    action_values = np.asarray(action, dtype=np.float64).reshape(-1)
    if action_values.size != 1:
        raise ValueError("Pendulum action must contain exactly one scalar")
    error = angle_normalize(theta - float(target_angle))
    return -(error**2 + 0.1 * theta_dot**2 + 0.001 * float(action_values[0]) ** 2)


class TargetAngleContext(gym.Wrapper):
    """Replace only reward and expose a fixed-per-episode target-angle context."""

    def __init__(
        self,
        env: gym.Env,
        goals: Mapping[int, float],
        mode: str,
        normalization: tuple[float, float],
    ) -> None:
        super().__init__(env)
        if mode not in {"hidden", "oracle"}:
            raise ValueError("mode must be 'hidden' or 'oracle'")
        if not goals:
            raise ValueError("At least one target angle is required")
        center, scale = map(float, normalization)
        if scale <= 0:
            raise ValueError("Target-angle normalization scale must be positive")
        self.goals = {int(key): float(value) for key, value in goals.items()}
        self.mode = mode
        self.context_center = center
        self.context_scale = scale
        state_space = env.observation_space["obs"]
        if not isinstance(state_space, gym.spaces.Box) or state_space.shape != (3,):
            raise TypeError("CARLPendulum state observation must be a three-value Box")
        low = np.asarray(state_space.low, dtype=np.float32).copy()
        high = np.asarray(state_space.high, dtype=np.float32).copy()
        if mode == "oracle":
            low = np.concatenate((low, np.array([-np.inf], dtype=np.float32)))
            high = np.concatenate((high, np.array([np.inf], dtype=np.float32)))
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self._target_angle: float | None = None
        self._context_id: int | None = None
        self._last_observation: Any = None

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        context_id = int(info["context_id"])
        if context_id not in self.goals:
            raise RuntimeError(f"CARL selected unknown target-angle context {context_id}")
        self._context_id = context_id
        self._target_angle = self.goals[context_id]
        self._last_observation = observation
        return self._observation(observation), self._context_info(info)

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._target_angle is None or self._last_observation is None:
            raise RuntimeError("reset() must be called before step()")
        observation, original_reward, terminated, truncated, info = self.env.step(action)
        if int(info["context_id"]) != self._context_id:
            raise RuntimeError("Target angle changed within an episode")
        # Preserve the native task as a literal identity case.  Reconstructing
        # theta from float32 cos/sin observations is numerically faithful but
        # can differ from Pendulum's state-based reward by a few ulps.
        reward = (
            float(original_reward)
            if self._target_angle == 0.0
            else target_reward(self._last_observation, action, self._target_angle)
        )
        self._last_observation = observation
        return self._observation(observation), reward, terminated, truncated, self._context_info(info)

    def _observation(self, observation: Any) -> np.ndarray:
        state = pendulum_state(observation)
        if self.mode == "hidden":
            return state
        return np.concatenate(
            (state, np.array([self.normalized_target_angle], dtype=np.float32))
        ).astype(np.float32, copy=False)

    def _context_info(self, info: Mapping[str, Any]) -> dict[str, Any]:
        enriched = dict(info)
        enriched.update(
            {
                "context_id": self._context_id,
                "target_angle": self.target_angle,
                "normalized_target_angle": self.normalized_target_angle,
            }
        )
        return enriched

    @property
    def target_angle(self) -> float:
        if self._target_angle is None:
            raise RuntimeError("No target angle is active before reset()")
        return self._target_angle

    @property
    def normalized_target_angle(self) -> float:
        return (self.target_angle - self.context_center) / self.context_scale

    @property
    def active_context(self) -> dict[str, float]:
        return {"target_angle": self.target_angle}

    @property
    def active_context_id(self) -> int | None:
        return self._context_id


def make_goal_pendulum_env(
    goals: Mapping[int, float],
    mode: str,
    seed: int,
    *,
    normalization: tuple[float, float],
    static_context: bool = False,
) -> TargetAngleContext:
    """Create CARLPendulum with identical dynamics in every goal context."""
    if not goals:
        raise ValueError("At least one target angle is required")
    defaults = CARLPendulum.get_default_context()
    carl_contexts = {int(context_id): dict(defaults) for context_id in goals}
    base_env = gym.make("Pendulum-v1", render_mode="rgb_array")
    carl_env = CARLPendulum(
        env=base_env,
        contexts=carl_contexts,
        obs_context_features=[],
        obs_context_as_dict=False,
        context_selector=StaticSelector if static_context else None,
    )
    selector_goals = {index: float(goal) for index, goal in enumerate(goals.values())}
    env = TargetAngleContext(carl_env, selector_goals, mode, normalization)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env
