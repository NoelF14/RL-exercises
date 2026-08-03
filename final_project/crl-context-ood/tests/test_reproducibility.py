from __future__ import annotations

import numpy as np

from crl_ood.environments.context_splits import (
    build_context_splits,
    context_normalization,
)
from crl_ood.environments.factory import make_pendulum_env
from crl_ood.utils.seeding import seed_everything


def _short_rollout(smoke_config):
    seed_everything(101)
    contexts = build_context_splits(
        "dt", smoke_config["environment"]["splits"], seed=23
    )["train"]
    env = make_pendulum_env(
        contexts,
        "dt",
        "hidden",
        seed=101,
        context_normalization=context_normalization(contexts, "dt"),
    )
    observation, info = env.reset(seed=101)
    trajectory = [(observation.copy(), info["context_id"], 0.0)]
    for action in (
        np.array([-0.5], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        np.array([0.75], dtype=np.float32),
        np.array([-1.0], dtype=np.float32),
    ):
        observation, reward, terminated, truncated, info = env.step(action)
        trajectory.append((observation.copy(), info["context_id"], float(reward)))
        assert not (terminated or truncated)
    env.close()
    return trajectory


def test_short_rollout_reproducible_with_fixed_seed(smoke_config):
    first = _short_rollout(smoke_config)
    second = _short_rollout(smoke_config)
    for first_step, second_step in zip(first, second, strict=True):
        np.testing.assert_array_equal(first_step[0], second_step[0])
        assert first_step[1:] == second_step[1:]
