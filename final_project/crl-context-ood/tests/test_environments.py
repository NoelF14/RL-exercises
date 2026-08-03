from __future__ import annotations

import numpy as np
import pytest

from crl_ood.environments.context_splits import build_context_splits, carl_feature_key
from crl_ood.environments.factory import make_pendulum_env


def test_round_robin_context_assignment_is_deterministic(smoke_config):
    contexts = build_context_splits(
        "gravity", smoke_config["environment"]["splits"], seed=7
    )["train"]
    observed_sequences = []
    for _ in range(2):
        env = make_pendulum_env(contexts, "gravity", "hidden", seed=11)
        sequence = [env.reset(seed=100 + episode)[1]["context_id"] for episode in range(6)]
        observed_sequences.append(sequence)
        env.close()

    assert observed_sequences[0] == [0, 1, 2, 0, 1, 2]
    assert observed_sequences[0] == observed_sequences[1]


@pytest.mark.parametrize(
    ("mode", "shape"), [("hidden", (3,)), ("oracle", (4,))]
)
def test_observation_shapes(smoke_config, mode, shape):
    context = build_context_splits(
        "length", smoke_config["environment"]["splits"], seed=0
    )["train"]
    env = make_pendulum_env({0: context[0]}, "length", mode, seed=3, static_context=True)
    observation, _ = env.reset(seed=3)

    assert env.observation_space.shape == shape
    assert observation.shape == shape
    assert env.observation_space.contains(observation)
    env.close()


@pytest.mark.parametrize("feature", ["gravity", "length", "dt"])
def test_hidden_and_oracle_evaluate_identical_contexts(smoke_config, feature):
    contexts = build_context_splits(
        feature, smoke_config["environment"]["splits"], seed=19
    )["ood_high"]
    key = carl_feature_key(feature)

    for context_id, context in contexts.items():
        hidden = make_pendulum_env(
            {context_id: context}, feature, "hidden", seed=5, static_context=True
        )
        oracle = make_pendulum_env(
            {context_id: context}, feature, "oracle", seed=5, static_context=True
        )
        hidden_obs, _ = hidden.reset(seed=1234 + context_id)
        oracle_obs, _ = oracle.reset(seed=1234 + context_id)

        assert hidden.active_context[key] == oracle.active_context[key] == context[key]
        np.testing.assert_array_equal(hidden_obs, oracle_obs[:3])
        assert oracle_obs[-1] == pytest.approx(context[key])
        hidden.close()
        oracle.close()
