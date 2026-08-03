from __future__ import annotations

import math

import numpy as np
import pytest

from crl_ood.goal_pilot.environment import (
    angle_normalize,
    make_goal_pendulum_env,
    target_reward,
)


def _state(theta: float, theta_dot: float = 0.0) -> dict[str, np.ndarray]:
    return {"obs": np.array([math.cos(theta), math.sin(theta), theta_dot], dtype=np.float32)}


def test_target_reward_correct_at_known_angles():
    assert target_reward(_state(0.3, 2.0), np.array([1.5]), 0.3) == pytest.approx(-(0.4 + 0.00225))
    assert target_reward(_state(0.0), np.array([0.0]), 0.6) == pytest.approx(-0.36)


def test_standard_angle_wrapping_near_pi_boundaries():
    assert angle_normalize(math.pi) == pytest.approx(-math.pi)
    assert angle_normalize(-math.pi) == pytest.approx(-math.pi)
    assert angle_normalize(math.pi + 1e-7) == pytest.approx(-math.pi + 1e-7)
    wrapped_error_reward = target_reward(_state(math.pi - 0.01), [0.0], -math.pi + 0.01)
    assert wrapped_error_reward == pytest.approx(-(0.02**2), abs=2e-8)


def test_target_changes_reward_but_not_transition_or_episode_flags():
    left = make_goal_pendulum_env({0: -0.6}, "hidden", 7, normalization=(0.0, 0.6), static_context=True)
    right = make_goal_pendulum_env({0: 0.6}, "hidden", 7, normalization=(0.0, 0.6), static_context=True)
    left_obs, _ = left.reset(seed=91)
    right_obs, _ = right.reset(seed=91)
    np.testing.assert_array_equal(left_obs, right_obs)
    action = np.array([0.4], dtype=np.float32)
    left_next, left_reward, left_terminated, left_truncated, _ = left.step(action)
    right_next, right_reward, right_terminated, right_truncated, _ = right.step(action)
    np.testing.assert_array_equal(left_next, right_next)
    assert (left_terminated, left_truncated) == (right_terminated, right_truncated)
    assert left_reward != right_reward
    left.close(); right.close()


def test_target_constant_within_episode_and_persisted_in_info():
    env = make_goal_pendulum_env({4: -0.6, 9: 0.6}, "oracle", 3, normalization=(0.0, 0.6))
    observation, info = env.reset(seed=3)
    target = info["target_angle"]
    normalized = info["normalized_target_angle"]
    assert observation[-1] == pytest.approx(normalized)
    for _ in range(20):
        observation, _, terminated, truncated, info = env.step(np.array([0.0], dtype=np.float32))
        assert info["target_angle"] == target
        assert info["normalized_target_angle"] == normalized
        assert observation[-1] == pytest.approx(normalized)
        assert not (terminated or truncated)
    env.close()


def test_hidden_oracle_dimensions_and_training_extrema_normalization():
    hidden = make_goal_pendulum_env({0: -1.0}, "hidden", 0, normalization=(0.0, 0.6), static_context=True)
    oracle = make_goal_pendulum_env({0: -1.0}, "oracle", 0, normalization=(0.0, 0.6), static_context=True)
    hidden_obs, hidden_info = hidden.reset(seed=8)
    oracle_obs, oracle_info = oracle.reset(seed=8)
    assert hidden.observation_space.shape == hidden_obs.shape == (3,)
    assert oracle.observation_space.shape == oracle_obs.shape == (4,)
    np.testing.assert_array_equal(hidden_obs, oracle_obs[:3])
    assert oracle_obs[-1] == pytest.approx(-1.0 / 0.6)
    assert hidden_info["target_angle"] == oracle_info["target_angle"] == -1.0
    assert oracle.observation_space.contains(oracle_obs)
    hidden.close(); oracle.close()
