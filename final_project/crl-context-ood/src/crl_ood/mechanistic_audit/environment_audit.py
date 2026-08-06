"""Run target-angle mechanics and CARLPendulum reset audits."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from carl.context.selection import StaticSelector
from carl.envs import CARLPendulum

from crl_ood.goal_pilot.environment import (
    TargetAngleContext,
    angle_normalize,
    pendulum_state,
    target_reward,
)

PILOT_GOALS = (-0.6, 0.0, 0.6)
ABS_TOL = 2e-7


def make_original_carl_pendulum() -> CARLPendulum:
    """Construct the exact CARL/base-environment stack used by the pilot."""
    return CARLPendulum(
        env=gym.make("Pendulum-v1", render_mode="rgb_array"),
        contexts={0: dict(CARLPendulum.get_default_context())},
        obs_context_features=[],
        obs_context_as_dict=False,
        context_selector=StaticSelector,
    )


def run_environment_audit(
    output_dir: str | Path, *, reset_samples: int = 10_000,
    reset_seed_offset: int = 4_000_000,
) -> dict[str, Any]:
    """Run all environment checks and persist machine-readable evidence."""
    if reset_samples < 10_000:
        raise ValueError("The real reset audit requires at least 10,000 resets")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    findings = {
        "goal_zero_equivalence": audit_goal_zero_equivalence(),
        "angle_extraction": audit_angle_extraction(),
        "reward": audit_reward_formula_and_mirror(),
        "dynamics": audit_dynamics_symmetry(),
        "reset_distribution": audit_reset_distribution(
            output, samples=reset_samples, seed_offset=reset_seed_offset
        ),
    }
    findings["all_mechanistic_invariants_pass"] = all((
        findings["goal_zero_equivalence"]["passed"],
        findings["angle_extraction"]["passed"],
        findings["reward"]["formula_passed"],
        findings["reward"]["mirror_passed"],
        findings["dynamics"]["mirror_passed"],
    ))
    (output / "environment_audit_findings.json").write_text(
        json.dumps(findings, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return findings


def audit_goal_zero_equivalence() -> dict[str, Any]:
    """Compare original CARL and target-zero step-for-step, including timing."""
    original = make_original_carl_pendulum()
    wrapped_base = make_original_carl_pendulum()
    wrapped = TargetAngleContext(wrapped_base, {0: 0.0}, "hidden", (0.0, 0.6))
    max_obs = max_state = max_reward = 0.0
    steps = episodes = 0
    try:
        actions = [np.array([2.0 * math.sin(i * 0.37)], dtype=np.float32) for i in range(200)]
        for seed in (3, 17, 91, 1234):
            original_obs, _ = original.reset(seed=seed)
            wrapped_obs, _ = wrapped.reset(seed=seed)
            max_obs = max(max_obs, _max_abs(pendulum_state(original_obs), wrapped_obs))
            max_state = max(max_state, _max_abs(_state(original), _state(wrapped_base)))
            original_length = wrapped_length = 0
            for action in actions:
                original_next = original.step(action)
                wrapped_next = wrapped.step(action)
                original_length += 1
                wrapped_length += 1
                oo, oreward, oterm, otrunc, _ = original_next
                wo, wreward, wterm, wtrunc, _ = wrapped_next
                max_obs = max(max_obs, _max_abs(pendulum_state(oo), wo))
                max_state = max(max_state, _max_abs(_state(original), _state(wrapped_base)))
                max_reward = max(max_reward, abs(float(oreward) - wreward))
                if (oterm, otrunc) != (wterm, wtrunc):
                    raise AssertionError("Target-zero changed episode flags")
                if original_length != wrapped_length:
                    raise AssertionError("Target-zero changed episode length")
                steps += 1
                if oterm or otrunc:
                    break
            episodes += 1
    finally:
        original.close()
        wrapped.close()
    timing = _reward_timing()
    return {
        "passed": max_obs <= ABS_TOL and max_state <= ABS_TOL
        and max_reward == 0.0 and timing["convention"] == "pre_transition",
        "episodes_checked": episodes,
        "steps_checked": steps,
        "max_observation_abs_error": max_obs,
        "max_underlying_state_abs_error": max_state,
        "max_reward_abs_error": max_reward,
        "reward_bit_identity": max_reward == 0.0,
        "reward_timing": timing,
        "episode_lengths_equal": True,
        "termination_flags_equal": True,
    }


def _reward_timing() -> dict[str, Any]:
    env = make_original_carl_pendulum()
    try:
        env.reset(seed=77)
        before = np.array([0.73, -0.41], dtype=np.float64)
        env.env.unwrapped.state = before.copy()
        action = np.array([1.2], dtype=np.float32)
        _, observed, _, _, _ = env.step(action)
        after = _state(env)
        pre = _formula(before[0], before[1], float(action[0]), 0.0)
        post = _formula(after[0], after[1], float(action[0]), 0.0)
        pre_error = abs(float(observed) - pre)
        post_error = abs(float(observed) - post)
        return {
            "convention": "pre_transition" if pre_error < post_error else "post_transition",
            "observed_reward": float(observed),
            "pre_transition_formula": pre,
            "post_transition_formula": post,
            "pre_abs_error": pre_error,
            "post_abs_error": post_error,
        }
    finally:
        env.close()


def audit_angle_extraction() -> dict[str, Any]:
    """Check observation index semantics against the actual unwrapped state."""
    env = make_original_carl_pendulum()
    max_error = 0.0
    swapped_guard = False
    checks = 0
    try:
        for seed in (0, 1, 2, 999):
            observation, _ = env.reset(seed=seed)
            error, swapped = _angle_errors(observation, _state(env)[0])
            max_error, swapped_guard, checks = max(max_error, error), swapped_guard or swapped > 0.1, checks + 1
        for theta in (-math.pi + 1e-8, -2.2, -0.7, 0.0, 0.37, 1.8,
                      math.pi - 1e-8, math.pi + 1e-8):
            env.env.unwrapped.state = np.array([theta, -0.25], dtype=np.float64)
            observation = {"obs": env.env.unwrapped._get_obs()}
            error, swapped = _angle_errors(observation, theta)
            max_error, swapped_guard, checks = max(max_error, error), swapped_guard or swapped > 0.1, checks + 1
        env.reset(seed=456)
        rng = np.random.default_rng(8181)
        for _ in range(128):
            observation, _, _, _, _ = env.step(np.array([rng.uniform(-2, 2)], dtype=np.float32))
            error, swapped = _angle_errors(observation, _state(env)[0])
            max_error, swapped_guard, checks = max(max_error, error), swapped_guard or swapped > 0.1, checks + 1
    finally:
        env.close()
    return {
        "passed": max_error <= ABS_TOL and swapped_guard,
        "checks": checks,
        "max_wrapped_angle_abs_error": max_error,
        "observation_order": ["cos_theta", "sin_theta", "theta_dot"],
        "swapped_sin_cos_guard_triggered": swapped_guard,
    }


def _angle_errors(observation: Any, theta: float) -> tuple[float, float]:
    values = pendulum_state(observation)
    expected = angle_normalize(float(theta))
    actual = math.atan2(float(values[1]), float(values[0]))
    swapped = math.atan2(float(values[0]), float(values[1]))
    return abs(angle_normalize(actual - expected)), abs(angle_normalize(swapped - expected))


def audit_reward_formula_and_mirror() -> dict[str, Any]:
    """Check sign, wrapping, and the complete mirror relation."""
    max_formula = max_mirror = 0.0
    checks = 0
    epsilon = 1e-7
    for goal in PILOT_GOALS:
        angles = {-math.pi + epsilon, -math.pi - epsilon, -epsilon, 0.0, epsilon,
                  math.pi - epsilon, math.pi + epsilon, goal - epsilon, goal, goal + epsilon}
        for theta in angles:
            for velocity in (-1.7, 0.0, 1.3):
                for action in (-2.0, -0.4, 0.0, 0.9, 2.0):
                    observation = {"obs": np.array([math.cos(theta), math.sin(theta), velocity], dtype=np.float32)}
                    actual = target_reward(observation, [action], goal)
                    expected = _formula(theta, velocity, action, goal)
                    mirrored_obs = {"obs": np.array([math.cos(-theta), math.sin(-theta), -velocity], dtype=np.float32)}
                    mirrored = target_reward(mirrored_obs, [-action], -goal)
                    max_formula = max(max_formula, abs(actual - expected))
                    max_mirror = max(max_mirror, abs(actual - mirrored))
                    if actual > 1e-12:
                        raise AssertionError("Target reward must be non-positive")
                    checks += 1
    return {
        "formula_passed": max_formula <= 1e-6,
        "mirror_passed": max_mirror <= 1e-6,
        "checks": checks,
        "max_formula_abs_error": max_formula,
        "max_mirror_abs_error": max_mirror,
        "targets": list(PILOT_GOALS),
    }


def audit_dynamics_symmetry() -> dict[str, Any]:
    """Compare mirrored transitions, independently from reset symmetry."""
    left, right = make_original_carl_pendulum(), make_original_carl_pendulum()
    max_state = max_obs = 0.0
    cases = ((-math.pi + 1e-6, -0.9, -1.7), (-1.2, 0.4, 0.8),
             (-0.1, 7.9, 2.0), (0.0, 0.0, 0.0), (0.8, -7.9, -2.0),
             (math.pi - 1e-6, 1.1, 1.3))
    try:
        left.reset(seed=12); right.reset(seed=12)
        for theta, velocity, action in cases:
            left.env.unwrapped.state = np.array([theta, velocity], dtype=np.float64)
            right.env.unwrapped.state = np.array([-theta, -velocity], dtype=np.float64)
            lo, _, lt, lu, _ = left.step(np.array([action], dtype=np.float32))
            ro, _, rt, ru, _ = right.step(np.array([-action], dtype=np.float32))
            max_state = max(max_state, _max_abs(_state(left), -_state(right)))
            lv, rv = pendulum_state(lo), pendulum_state(ro)
            max_obs = max(max_obs, _max_abs(lv, [rv[0], -rv[1], -rv[2]]))
            if (lt, lu) != (rt, ru):
                raise AssertionError("Mirrored dynamics changed episode flags")
    finally:
        left.close(); right.close()
    return {
        "mirror_passed": max_state <= ABS_TOL and max_obs <= ABS_TOL,
        "transitions_checked": len(cases),
        "max_underlying_state_mirror_abs_error": max_state,
        "max_observation_mirror_abs_error": max_obs,
        "reset_distribution_tested_separately": True,
    }


def audit_reset_distribution(output: Path, *, samples: int, seed_offset: int) -> dict[str, Any]:
    """Sample real CARL resets and write summaries, distances, and histograms."""
    env = make_original_carl_pendulum()
    thetas: list[float] = []
    velocities: list[float] = []
    distances: list[dict[str, Any]] = []
    try:
        for index in range(samples):
            seed = seed_offset + index
            observation, _ = env.reset(seed=seed)
            values = pendulum_state(observation)
            theta = math.atan2(float(values[1]), float(values[0]))
            velocity = float(values[2])
            thetas.append(theta); velocities.append(velocity)
            for goal in PILOT_GOALS:
                distances.append({"reset_index": index, "reset_seed": seed,
                                  "target_angle": goal, "theta": theta,
                                  "absolute_wrapped_distance": abs(angle_normalize(theta - goal))})
    finally:
        env.close()
    summaries = [_distribution_row("theta", thetas), _distribution_row("theta_dot", velocities)]
    _write_csv(output / "reset_distribution_summary.csv", summaries)
    _write_csv(output / "initial_distance_to_goal.csv", distances)
    _histogram(thetas, output / "reset_theta_histogram.png", "Initial theta", "radians")
    _histogram(velocities, output / "reset_theta_dot_histogram.png", "Initial theta_dot", "radians / second")
    theta_row, velocity_row = summaries
    return {
        "samples": samples,
        "seed_offset": seed_offset,
        "theta_negative_fraction": theta_row["negative_fraction"],
        "theta_positive_fraction": theta_row["positive_fraction"],
        "theta_dot_negative_fraction": velocity_row["negative_fraction"],
        "theta_dot_positive_fraction": velocity_row["positive_fraction"],
        "asymmetric": abs(theta_row["negative_fraction"] - 0.5) > 0.05
        or abs(velocity_row["negative_fraction"] - 0.5) > 0.05,
        "observed_support": {"theta": [theta_row["min"], theta_row["max"]],
                             "theta_dot": [velocity_row["min"], velocity_row["max"]]},
    }


def _distribution_row(name: str, values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {"variable": name, "samples": len(values), "mean": float(np.mean(array)),
            "std": float(np.std(array)), "min": float(np.min(array)),
            "q05": float(np.quantile(array, .05)), "median": float(np.median(array)),
            "q95": float(np.quantile(array, .95)), "max": float(np.max(array)),
            "negative_fraction": float(np.mean(array < 0)),
            "zero_fraction": float(np.mean(array == 0)),
            "positive_fraction": float(np.mean(array > 0))}


def _histogram(values: list[float], path: Path, title: str, xlabel: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.hist(values, bins=50, color="#4472C4", edgecolor="white", linewidth=.3)
    axis.set(title=title, xlabel=xlabel, ylabel="reset count")
    figure.tight_layout(); figure.savefig(path, dpi=160); plt.close(figure)


def _formula(theta: float, theta_dot: float, action: float, goal: float) -> float:
    error = angle_normalize(theta - goal)
    return -(error**2 + .1 * theta_dot**2 + .001 * action**2)


def _state(env: Any) -> np.ndarray:
    return np.asarray(env.env.unwrapped.state, dtype=np.float64).copy()


def _max_abs(left: Any, right: Any) -> float:
    return float(np.max(np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/goal_pilot_mechanistic_audit/environment"))
    parser.add_argument("--reset-samples", type=int, default=10_000)
    parser.add_argument("--reset-seed-offset", type=int, default=4_000_000)
    args = parser.parse_args()
    run_environment_audit(args.output_dir, reset_samples=args.reset_samples,
                          reset_seed_offset=args.reset_seed_offset)


if __name__ == "__main__":
    main()
