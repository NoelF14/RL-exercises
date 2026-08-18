"""Frozen history-latent observation wrapper for CARL CartPole."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from crl_ood.environments.factory import make_env
from crl_ood.pointrobot_encoders.dataset import transition_features
from crl_ood.pointrobot_encoders.training import load_frozen_checkpoint


class FrozenCartPoleHistoryObservation(gym.Wrapper):
    """Append a deterministic frozen history latent to hidden CartPole state."""

    def __init__(
        self,
        env: gym.Env,
        checkpoint: str | Path,
        expected_method: str,
        expected_dataset_checksum: str,
    ) -> None:
        super().__init__(env)

        self.encoder, self.checkpoint = load_frozen_checkpoint(
            checkpoint,
            expected_method=expected_method,
            expected_dataset_checksum=expected_dataset_checksum,
        )

        spec = self.checkpoint["config"]["encoder"]

        self.history_length = int(spec["history_length"])
        self.transition_dim = int(spec["transition_dim"])
        self.latent_dim = int(spec["latent_dim"])

        if self.transition_dim != 10:
            raise ValueError(
                f"CartPole encoder must use transition_dim=10, got {self.transition_dim}"
            )

        normalization = self.checkpoint["normalization"]

        self._mean = np.asarray(
            normalization["mean"],
            dtype=np.float32,
        )
        self._std = np.asarray(
            normalization["std"],
            dtype=np.float32,
        )

        if len(self._mean) != self.transition_dim:
            raise ValueError("checkpoint normalization dimension mismatch")

        self._history: deque[np.ndarray] = deque(
            maxlen=self.history_length
        )

        self._state: np.ndarray | None = None

        state_space = env.observation_space

        if not isinstance(state_space, gym.spaces.Box):
            raise TypeError("hidden CartPole observation must be a Box")

        state_low = np.asarray(
            state_space.low,
            dtype=np.float32,
        ).reshape(-1)

        state_high = np.asarray(
            state_space.high,
            dtype=np.float32,
        ).reshape(-1)

        if len(state_low) != 4:
            raise ValueError(
                f"expected 4-D CartPole state, got {len(state_low)}"
            )

        self.observation_space = gym.spaces.Box(
            low=np.concatenate(
                (
                    state_low,
                    np.full(
                        self.latent_dim,
                        -np.inf,
                        dtype=np.float32,
                    ),
                )
            ),
            high=np.concatenate(
                (
                    state_high,
                    np.full(
                        self.latent_dim,
                        np.inf,
                        dtype=np.float32,
                    ),
                )
            ),
            dtype=np.float32,
        )

    @property
    def checkpoint_checksum(self) -> str:
        return str(
            self.checkpoint["checkpoint_checksum"]
        )

    def reset(
        self,
        **kwargs: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        state, info = self.env.reset(**kwargs)

        state = np.asarray(
            state,
            dtype=np.float32,
        ).reshape(-1)

        self._history.clear()
        self._state = state.copy()

        return self._observation(state), info

    def step(
        self,
        action: Any,
    ) -> tuple[
        np.ndarray,
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        if self._state is None:
            raise RuntimeError(
                "reset must precede step"
            )

        next_state, reward, terminated, truncated, info = (
            self.env.step(action)
        )

        next_state = np.asarray(
            next_state,
            dtype=np.float32,
        ).reshape(-1)

        action_array = np.asarray(
            action,
        ).reshape(-1)

        if action_array.size != 1:
            raise ValueError(
                f"expected scalar CartPole action, got shape {np.asarray(action).shape}"
            )

        action_feature = np.asarray(
            [[float(action_array[0])]],
            dtype=np.float32,
        )

        completed = transition_features(
            self._state[None],
            action_feature,
            np.asarray(
                [reward],
                dtype=np.float32,
            ),
            next_state[None],
        )[0]

        normalized = (
            completed - self._mean
        ) / self._std

        self._history.append(
            normalized.astype(np.float32)
        )

        self._state = next_state.copy()

        return (
            self._observation(next_state),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def _observation(
        self,
        state: np.ndarray,
    ) -> np.ndarray:
        history = np.zeros(
            (
                1,
                self.history_length,
                self.transition_dim,
            ),
            dtype=np.float32,
        )

        length = len(self._history)

        if length:
            history[0, :length] = np.asarray(
                self._history,
                dtype=np.float32,
            )

        lengths = torch.as_tensor(
            [length],
            dtype=torch.long,
        )

        mask = (
            torch.arange(
                self.history_length
            )[None, :]
            < lengths[:, None]
        )

        with torch.no_grad():
            latent = self.encoder.encode(
                torch.from_numpy(history),
                lengths,
                mask,
                deterministic=True,
            )[0].cpu().numpy()

        return np.concatenate(
            (
                np.asarray(
                    state,
                    dtype=np.float32,
                ),
                latent.astype(np.float32),
            )
        )


def make_cartpole_history_env(
    contexts,
    *,
    feature: str,
    seed: int,
    context_normalization: tuple[float, float],
    checkpoint: str | Path,
    method: str,
    dataset_checksum: str,
    static_context: bool = False,
) -> FrozenCartPoleHistoryObservation:
    """Create hidden CARL CartPole with a frozen learned history representation."""

    base = make_env(
        "cartpole",
        contexts,
        feature,
        "hidden",
        seed,
        context_normalization=context_normalization,
        static_context=static_context,
    )

    return FrozenCartPoleHistoryObservation(
        base,
        checkpoint,
        expected_method=method,
        expected_dataset_checksum=dataset_checksum,
    )
