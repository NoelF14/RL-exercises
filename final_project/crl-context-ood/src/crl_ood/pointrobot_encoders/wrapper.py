"""Frozen-latent PointRobot observation wrapper with completed-transition history."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
import torch

from crl_ood.pointrobot_encoders.dataset import transition_features
from crl_ood.pointrobot_encoders.training import load_frozen_checkpoint
from crl_ood.pointrobot_gate.environment import DenseSemiCirclePointRobot


class FrozenHistoryObservation(gym.Wrapper[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
    """Append a deterministic frozen latent; each wrapper owns independent history."""

    def __init__(self, env: DenseSemiCirclePointRobot, checkpoint: str | Path,
                 expected_method: str, expected_dataset_checksum: str) -> None:
        if env.observation_mode != "hidden":
            raise ValueError("learned history wrappers require a hidden-state base environment")
        super().__init__(env)
        self.encoder, self.checkpoint = load_frozen_checkpoint(
            checkpoint, expected_method=expected_method,
            expected_dataset_checksum=expected_dataset_checksum)
        spec = self.checkpoint["config"]["encoder"]
        self.history_length = int(spec["history_length"])
        self.latent_dim = int(spec["latent_dim"])
        norm = self.checkpoint["normalization"]
        self._mean = np.asarray(norm["mean"], dtype=np.float32)
        self._std = np.asarray(norm["std"], dtype=np.float32)
        self._history: deque[np.ndarray] = deque(maxlen=self.history_length)
        self._state: np.ndarray | None = None
        limit = float(env.position_limit)
        self.observation_space = gym.spaces.Box(
            low=np.r_[[-limit, -limit], np.full(self.latent_dim, -np.inf)],
            high=np.r_[[limit, limit], np.full(self.latent_dim, np.inf)], dtype=np.float32)

    @property
    def checkpoint_checksum(self) -> str:
        return str(self.checkpoint["checkpoint_checksum"])

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        state, info = self.env.reset(**kwargs)
        self._history.clear(); self._state = state.copy()
        return self._observation(state), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("reset must precede step")
        next_state, reward, terminated, truncated, info = self.env.step(action)
        completed = transition_features(self._state[None], np.asarray(action, dtype=np.float32)[None],
                                        np.asarray([reward], dtype=np.float32), next_state[None])[0]
        self._history.append((completed - self._mean) / self._std)
        self._state = next_state.copy()
        return self._observation(next_state), reward, terminated, truncated, info

    def _observation(self, state: np.ndarray) -> np.ndarray:
        history = np.zeros((1, self.history_length, 7), dtype=np.float32)
        length = len(self._history)
        if length:
            history[0, :length] = np.asarray(self._history)
        lengths = torch.as_tensor([length], dtype=torch.long)
        mask = torch.arange(self.history_length)[None, :] < lengths[:, None]
        with torch.no_grad():
            latent = self.encoder.encode(torch.from_numpy(history), lengths, mask, deterministic=True)[0].numpy()
        return np.concatenate((np.asarray(state, dtype=np.float32), latent.astype(np.float32)))


def make_policy_env(method: str, goal_angles: float | Sequence[float], env_kwargs: dict[str, Any],
                    *, checkpoint: str | Path | None = None,
                    dataset_checksum: str | None = None) -> gym.Env:
    if method == "no_context":
        return DenseSemiCirclePointRobot(goal_angles, "hidden", **env_kwargs)
    if method == "oracle":
        return DenseSemiCirclePointRobot(goal_angles, "oracle", **env_kwargs)
    if method not in {"vae", "contrastive", "contrastive_alternative"}:
        raise ValueError(f"unknown downstream method {method!r}")
    if checkpoint is None or not dataset_checksum:
        raise ValueError("learned methods require explicit checkpoint and dataset checksum")
    base = DenseSemiCirclePointRobot(goal_angles, "hidden", **env_kwargs)
    return FrozenHistoryObservation(base, checkpoint, method, dataset_checksum)
