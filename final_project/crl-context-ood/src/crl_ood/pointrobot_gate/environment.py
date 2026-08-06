"""Independent dense state-based Semi-Circle PointRobot environment.

Inspired by ContraBAR's Semi-Circle PointRobot domain, but independently
implemented here with dense post-transition reward and a fixed horizon.
Context changes reward only; dynamics never depend on the goal angle.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np


class DenseSemiCirclePointRobot(gym.Env[np.ndarray, np.ndarray]):
    """Two-dimensional point mass with a fixed-per-episode goal on a semicircle."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        goal_angles: float | Sequence[float],
        observation_mode: str = "hidden",
        *,
        goal_radius: float = 1.0,
        start_position: Sequence[float] = (0.0, 0.0),
        reset_noise: float = 0.0,
        step_scale: float = 0.1,
        position_limit: float = 1.5,
        horizon: int = 50,
        action_penalty: float = 0.01,
        success_threshold: float = 0.10,
    ) -> None:
        super().__init__()
        if observation_mode not in {"hidden", "oracle"}:
            raise ValueError("observation_mode must be 'hidden' or 'oracle'")
        values = [float(goal_angles)] if np.isscalar(goal_angles) else [float(x) for x in goal_angles]
        if not values:
            raise ValueError("At least one goal angle is required")
        if goal_radius <= 0 or step_scale <= 0 or position_limit <= 0 or horizon <= 0:
            raise ValueError("radius, step scale, position limit, and horizon must be positive")
        if reset_noise < 0 or action_penalty < 0 or success_threshold < 0:
            raise ValueError("noise, penalty, and threshold must be nonnegative")
        start = np.asarray(start_position, dtype=np.float64)
        if start.shape != (2,) or np.any(np.abs(start) > position_limit):
            raise ValueError("start_position must be a two-vector within the position limit")

        self.goal_angles = tuple(values)
        self.observation_mode = observation_mode
        self.goal_radius = float(goal_radius)
        self.start_position = start
        self.reset_noise = float(reset_noise)
        self.step_scale = float(step_scale)
        self.position_limit = float(position_limit)
        self.horizon = int(horizon)
        self.action_penalty = float(action_penalty)
        self.success_threshold = float(success_threshold)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        obs_limit = np.array([position_limit, position_limit], dtype=np.float32)
        if observation_mode == "oracle":
            low = np.concatenate((-obs_limit, np.array([-1.0, -1.0], dtype=np.float32)))
            high = np.concatenate((obs_limit, np.array([1.0, 1.0], dtype=np.float32)))
        else:
            low, high = -obs_limit, obs_limit
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self._next_goal_index = 0
        self._goal_angle: float | None = None
        self._position = start.copy()
        self._step_count = 0
        self._minimum_distance = float("inf")
        self._first_success_timestep: int | None = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if options and "goal_angle" in options:
            angle = float(options["goal_angle"])
            if not any(np.isclose(angle, item, atol=1e-12) for item in self.goal_angles):
                raise ValueError("options goal_angle is not configured for this environment")
            self._goal_angle = angle
        else:
            self._goal_angle = self.goal_angles[self._next_goal_index % len(self.goal_angles)]
            self._next_goal_index += 1
        noise = self.np_random.uniform(-self.reset_noise, self.reset_noise, size=2)
        self._position = np.clip(self.start_position + noise, -self.position_limit, self.position_limit)
        self._step_count = 0
        initial_distance = self.distance_to_goal
        self._minimum_distance = initial_distance
        self._first_success_timestep = 0 if self._is_success(initial_distance) else None
        # Context is intentionally absent from info in hidden mode. Evaluation
        # persists it from the immutable plan, never from the policy interface.
        return self._observation(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._goal_angle is None:
            raise RuntimeError("reset() must be called before step()")
        if self._step_count >= self.horizon:
            raise RuntimeError("Episode is complete; reset() is required after the fixed horizon")
        clipped_action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if clipped_action.shape != (2,):
            raise ValueError("action must have shape (2,)")
        # Reward timing is post-transition: use p_next after both clips.
        self._position = np.clip(
            self._position + self.step_scale * clipped_action,
            -self.position_limit,
            self.position_limit,
        )
        self._step_count += 1
        distance = self.distance_to_goal
        self._minimum_distance = min(self._minimum_distance, distance)
        if self._first_success_timestep is None and self._is_success(distance):
            self._first_success_timestep = self._step_count
        reward = -(distance**2 + self.action_penalty * float(np.dot(clipped_action, clipped_action)))
        truncated = self._step_count == self.horizon
        info: dict[str, Any] = {}
        if truncated:
            info = {
                "success": self._is_success(distance),
                "final_distance": distance,
                "minimum_distance": self._minimum_distance,
                "first_success_timestep": self._first_success_timestep,
            }
        return self._observation(), float(reward), False, truncated, info

    def _is_success(self, distance: float) -> bool:
        """Apply the declared inclusive threshold robustly at its float boundary."""
        return bool(distance <= self.success_threshold + 1e-12)

    def _observation(self) -> np.ndarray:
        state = self._position.astype(np.float32, copy=True)
        if self.observation_mode == "hidden":
            return state
        return np.concatenate((state, np.array([np.cos(self.goal_angle), np.sin(self.goal_angle)], dtype=np.float32)))

    @property
    def goal_angle(self) -> float:
        if self._goal_angle is None:
            raise RuntimeError("No goal is active before reset()")
        return self._goal_angle

    @property
    def goal_position(self) -> np.ndarray:
        return self.goal_radius * np.array([np.cos(self.goal_angle), np.sin(self.goal_angle)], dtype=np.float64)

    @property
    def position(self) -> np.ndarray:
        return self._position.copy()

    @property
    def distance_to_goal(self) -> float:
        return float(np.linalg.norm(self._position - self.goal_position))

    @property
    def environment_spec(self) -> dict[str, Any]:
        return {
            "name": "DenseSemiCirclePointRobot-v0",
            "state": "p=(x,y)",
            "action": "a in [-1,1]^2",
            "goal": "R*[cos(phi),sin(phi)]",
            "dynamics": "clip(p + step_scale*clip(a,-1,1), -position_limit, position_limit)",
            "reward": "-(||p_next-goal||^2 + action_penalty*||clip(a)||^2)",
            "reward_timing": "post_transition",
            "goal_radius": self.goal_radius,
            "start_position": self.start_position.tolist(),
            "reset_noise": self.reset_noise,
            "step_scale": self.step_scale,
            "position_limit": self.position_limit,
            "horizon": self.horizon,
            "action_penalty": self.action_penalty,
            "success_threshold": self.success_threshold,
            "early_termination": False,
            "attribution": "Inspired by ContraBAR Semi-Circle PointRobot; independently implemented for this project.",
        }
