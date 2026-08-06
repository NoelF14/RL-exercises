from __future__ import annotations

import math

import numpy as np
import pytest

from crl_ood.pointrobot_gate.environment import DenseSemiCirclePointRobot


def test_exact_dynamics_action_clipping_position_clipping_and_post_transition_reward():
    env = DenseSemiCirclePointRobot(0.0, "hidden", position_limit=0.15)
    observation, info = env.reset(seed=1)
    np.testing.assert_array_equal(observation, [0.0, 0.0]); assert info == {}
    next_observation, reward, terminated, truncated, info = env.step(np.array([2.0, -0.5]))
    np.testing.assert_allclose(next_observation, [0.1, -0.05])
    expected = -((0.1 - 1.0) ** 2 + (-0.05) ** 2 + 0.01 * (1.0 ** 2 + 0.5 ** 2))
    assert reward == pytest.approx(expected); assert not terminated and not truncated and info == {}
    clipped, _, _, _, _ = env.step(np.array([1.0, -1.0]))
    np.testing.assert_allclose(clipped, [0.15, -0.15])


def test_reward_uses_next_not_previous_position():
    env = DenseSemiCirclePointRobot(0.0)
    env.reset(seed=0)
    _, reward, _, _, _ = env.step(np.array([1.0, 0.0]))
    assert reward == pytest.approx(-(0.9**2 + 0.01))
    assert reward != pytest.approx(-(1.0**2 + 0.01))


def test_fixed_horizon_metrics_and_no_early_termination():
    env = DenseSemiCirclePointRobot(0.0)
    env.reset(seed=0)
    for step in range(1, 51):
        _, _, terminated, truncated, info = env.step(np.array([1.0 if step <= 10 else 0.0, 0.0]))
        assert not terminated; assert truncated == (step == 50)
        assert bool(info) == (step == 50)
    assert info["success"] is True
    assert info["final_distance"] == pytest.approx(0.0, abs=1e-7)
    assert info["minimum_distance"] == pytest.approx(0.0, abs=1e-7)
    assert info["first_success_timestep"] == 9
    with pytest.raises(RuntimeError, match="fixed horizon"):
        env.step(np.zeros(2))


def test_deterministic_seeding_and_constant_goal():
    a = DenseSemiCirclePointRobot([-0.6, 0.6], reset_noise=0.05)
    b = DenseSemiCirclePointRobot([-0.6, 0.6], reset_noise=0.05)
    ao, _ = a.reset(seed=18); bo, _ = b.reset(seed=18)
    np.testing.assert_array_equal(ao, bo); angle = a.goal_angle
    for _ in range(5):
        a.step(np.zeros(2)); assert a.goal_angle == angle


def test_goal_never_changes_dynamics_and_hidden_leaks_no_goal_info():
    left, right = DenseSemiCirclePointRobot(-0.6), DenseSemiCirclePointRobot(0.6)
    left_obs, left_info = left.reset(seed=4); right_obs, right_info = right.reset(seed=4)
    np.testing.assert_array_equal(left_obs, right_obs); assert left_info == right_info == {}
    action = np.array([0.3, -0.4])
    left_next, left_reward, _, _, left_info = left.step(action)
    right_next, right_reward, _, _, right_info = right.step(action)
    np.testing.assert_array_equal(left_next, right_next); assert left_reward != right_reward
    assert left_info == right_info == {}; assert left.observation_space.shape == (2,)


def test_oracle_contains_exactly_cosine_and_sine_not_raw_angle_or_id():
    hidden = DenseSemiCirclePointRobot(0.3, "hidden"); oracle = DenseSemiCirclePointRobot(0.3, "oracle")
    hidden_obs, _ = hidden.reset(seed=1); oracle_obs, _ = oracle.reset(seed=1)
    assert oracle_obs.shape == (4,); np.testing.assert_array_equal(oracle_obs[:2], hidden_obs)
    np.testing.assert_allclose(oracle_obs[2:], [math.cos(0.3), math.sin(0.3)])


def test_all_goals_equal_initial_distance_and_mirror_equivariance():
    distances=[]
    for goal in (-1.0,-0.6,0.0,0.6,1.0):
        env=DenseSemiCirclePointRobot(goal); env.reset(); distances.append(env.distance_to_goal)
    np.testing.assert_allclose(distances, np.ones(5))
    upper=DenseSemiCirclePointRobot(0.6); lower=DenseSemiCirclePointRobot(-0.6)
    upper.reset(); lower.reset()
    up,r_up,_,_,_=upper.step(np.array([0.4,0.7])); down,r_down,_,_,_=lower.step(np.array([0.4,-0.7]))
    np.testing.assert_allclose(down,[up[0],-up[1]]); assert r_down==pytest.approx(r_up)
