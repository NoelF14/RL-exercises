from __future__ import annotations

import numpy as np
import pytest

from crl_ood.environments.context_splits import (
    build_context_splits,
    carl_feature_key,
    context_normalization,
)
from crl_ood.environments.factory import make_pendulum_env
from crl_ood.evaluation.evaluate import build_evaluation_plan


def test_round_robin_context_assignment_is_deterministic(smoke_config):
    contexts = build_context_splits(
        "gravity", smoke_config["environment"]["splits"], seed=7
    )["train"]
    normalization = context_normalization(contexts, "gravity")
    observed_sequences = []
    for _ in range(2):
        env = make_pendulum_env(
            contexts,
            "gravity",
            "hidden",
            seed=11,
            context_normalization=normalization,
        )
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
    normalization = context_normalization(context, "length")
    env = make_pendulum_env(
        {0: context[0]},
        "length",
        mode,
        seed=3,
        context_normalization=normalization,
        static_context=True,
    )
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
    train_contexts = build_context_splits(
        feature, smoke_config["environment"]["splits"], seed=19
    )["train"]
    normalization = context_normalization(train_contexts, feature)
    key = carl_feature_key(feature)

    for context_id, context in contexts.items():
        hidden = make_pendulum_env(
            {context_id: context},
            feature,
            "hidden",
            seed=5,
            context_normalization=normalization,
            static_context=True,
        )
        oracle = make_pendulum_env(
            {context_id: context},
            feature,
            "oracle",
            seed=5,
            context_normalization=normalization,
            static_context=True,
        )
        hidden_obs, _ = hidden.reset(seed=1234 + context_id)
        oracle_obs, _ = oracle.reset(seed=1234 + context_id)

        assert hidden.active_context[key] == oracle.active_context[key] == context[key]
        np.testing.assert_array_equal(hidden_obs, oracle_obs[:3])
        expected = (context[key] - normalization[0]) / normalization[1]
        assert oracle_obs[-1] == pytest.approx(expected)
        hidden.close()
        oracle.close()


def test_hidden_and_oracle_evaluation_plans_are_paired(smoke_config):
    splits = build_context_splits(
        "gravity", smoke_config["environment"]["splits"], seed=19
    )
    hidden = build_evaluation_plan(smoke_config, "gravity", "hidden", 7, splits)
    oracle = build_evaluation_plan(smoke_config, "gravity", "oracle", 7, splits)
    paired_fields = (
        "seed",
        "context_feature",
        "context_value",
        "split",
        "context_id",
        "episode_index",
        "episode_seed",
    )
    assert [tuple(row[field] for field in paired_fields) for row in hidden] == [
        tuple(row[field] for field in paired_fields) for row in oracle
    ]


@pytest.mark.parametrize("feature", ["gravity", "length", "dt"])
def test_selected_context_changes_pendulum_dynamics(smoke_config, feature):
    splits = build_context_splits(
        feature, smoke_config["environment"]["splits"], seed=2
    )
    normalization = context_normalization(splits["train"], feature)
    low_context = splits["ood_low"][0]
    high_context = splits["ood_high"][0]
    observations = []
    for context in (low_context, high_context):
        env = make_pendulum_env(
            {0: context},
            feature,
            "hidden",
            seed=8,
            context_normalization=normalization,
            static_context=True,
        )
        initial, _ = env.reset(seed=99)
        next_observation, reward, _, _, _ = env.step(
            np.array([0.5], dtype=np.float32)
        )
        observations.append((initial, next_observation, reward))
        env.close()

    np.testing.assert_array_equal(observations[0][0], observations[1][0])
    assert not np.array_equal(observations[0][1], observations[1][1])


def test_context_is_constant_within_episode(smoke_config):
    splits = build_context_splits(
        "gravity", smoke_config["environment"]["splits"], seed=4
    )
    normalization = context_normalization(splits["train"], "gravity")
    context = splits["train"][0]
    env = make_pendulum_env(
        {0: context},
        "gravity",
        "oracle",
        seed=4,
        context_normalization=normalization,
        static_context=True,
    )
    observation, info = env.reset(seed=4)
    expected_context = observation[-1]
    expected_id = info["context_id"]
    for _ in range(20):
        observation, _, terminated, truncated, info = env.step(
            np.array([0.0], dtype=np.float32)
        )
        assert observation[-1] == expected_context
        assert info["context_id"] == expected_id
        assert not (terminated or truncated)
    env.close()
