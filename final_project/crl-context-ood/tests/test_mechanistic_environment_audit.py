from __future__ import annotations

import math

import numpy as np
import pytest

from crl_ood.goal_pilot.environment import pendulum_state, target_reward
from crl_ood.mechanistic_audit.environment_audit import (
    audit_angle_extraction,
    audit_dynamics_symmetry,
    audit_goal_zero_equivalence,
    audit_reward_formula_and_mirror,
)


def test_goal_zero_is_stepwise_identical_and_native_reward_is_pre_transition():
    result = audit_goal_zero_equivalence()
    assert result["passed"]
    assert result["reward_bit_identity"]
    assert result["reward_timing"]["convention"] == "pre_transition"
    assert result["reward_timing"]["pre_abs_error"] < result["reward_timing"]["post_abs_error"]
    assert result["steps_checked"] == 800


def test_real_carl_angle_extraction_and_swapped_index_guard():
    result = audit_angle_extraction()
    assert result["passed"]
    assert result["observation_order"] == ["cos_theta", "sin_theta", "theta_dot"]
    assert result["swapped_sin_cos_guard_triggered"]
    theta = 0.37
    observation = {"obs": np.array([math.cos(theta), math.sin(theta), 0.2], dtype=np.float32)}
    values = pendulum_state(observation)
    correct = math.atan2(float(values[1]), float(values[0]))
    swapped = math.atan2(float(values[0]), float(values[1]))
    assert correct == pytest.approx(theta, abs=1e-7)
    assert swapped != pytest.approx(theta, abs=0.1)


def test_reward_formula_sign_wrapping_and_mirror_grid():
    result = audit_reward_formula_and_mirror()
    assert result["formula_passed"]
    assert result["mirror_passed"]
    assert result["checks"] >= 400
    for goal in (-0.6, 0.0, 0.6):
        observation = {"obs": np.array([math.cos(goal), math.sin(goal), 0.0], dtype=np.float32)}
        assert target_reward(observation, [0.0], goal) == pytest.approx(0.0, abs=1e-14)


def test_underlying_dynamics_are_mirror_symmetric_separate_from_resets():
    result = audit_dynamics_symmetry()
    assert result["mirror_passed"]
    assert result["reset_distribution_tested_separately"]

