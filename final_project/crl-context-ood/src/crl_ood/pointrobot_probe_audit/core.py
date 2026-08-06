"""Pure numerical helpers for the PointRobot probe audit."""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

FORBIDDEN_INPUT_FIELDS = frozenset(
    {"goal_angle", "goal_cos", "goal_sin", "goal_x", "goal_y", "context_id", "split", "filename"}
)
TRANSITION_FIELDS = ("state", "action", "reward", "next_state")


def circular_angle_error(predicted: float, target: float) -> float:
    return abs((float(predicted) - float(target) + math.pi) % (2.0 * math.pi) - math.pi)


def normalize_prediction(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12:
        return np.array([1.0, 0.0], dtype=float)
    return vector / norm


def reward_equation_b(next_position: np.ndarray, action: np.ndarray, reward: float,
                      action_penalty: float, goal_radius: float = 1.0) -> float:
    """Rearrange dense reward into p_next.T @ g = b."""
    position = np.asarray(next_position, dtype=float)
    clipped_action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
    return float(
        (float(reward) + position @ position + goal_radius**2
         + action_penalty * (clipped_action @ clipped_action)) / 2.0
    )


def analytic_goal(history: list[dict[str, Any]], action_penalty: float,
                  goal_radius: float = 1.0, rank_tolerance: float = 1.0e-10) -> dict[str, Any]:
    positions = np.stack([np.asarray(item["next_state"], dtype=float) for item in history])
    b = np.asarray([
        reward_equation_b(item["next_state"], item["action"], item["reward"], action_penalty, goal_radius)
        for item in history
    ])
    rank = int(np.linalg.matrix_rank(positions, tol=rank_tolerance))
    singular = np.linalg.svd(positions, compute_uv=False)
    condition = float(np.inf if rank < 2 or singular[-1] <= rank_tolerance else singular[0] / singular[-1])
    estimate = np.linalg.pinv(positions, rcond=rank_tolerance) @ b
    normalized = normalize_prediction(estimate)
    return {
        "rank": rank,
        "condition_number": condition,
        "estimated_goal_cos": float(estimate[0]),
        "estimated_goal_sin": float(estimate[1]),
        "normalized_goal_cos": float(normalized[0]),
        "normalized_goal_sin": float(normalized[1]),
        "estimated_angle": float(math.atan2(normalized[1], normalized[0])),
        "geometrically_identifiable": rank == 2,
    }


def transition_history(row: dict[str, Any], history_length: int) -> list[dict[str, Any]]:
    actions = np.asarray(row["actions"], dtype=float)
    states = np.asarray(row["states"], dtype=float)
    rewards = np.asarray(row["rewards"], dtype=float)
    if history_length <= 0 or history_length > len(actions):
        raise ValueError("Invalid history length")
    start = len(actions) - history_length
    return [
        {"state": states[index].copy(), "action": actions[index].copy(),
         "reward": float(rewards[index]), "next_state": states[index + 1].copy()}
        for index in range(start, len(actions))
    ]


def raw_history_features(row: dict[str, Any], history_length: int) -> np.ndarray:
    history = transition_history(row, history_length)
    return np.concatenate([
        np.asarray([item["state"] for item in history]).reshape(-1),
        np.asarray([history[-1]["next_state"]]).reshape(-1),
        np.asarray([item["action"] for item in history]).reshape(-1),
        np.asarray([item["reward"] for item in history]),
    ])


def sequence_features(row: dict[str, Any], history_length: int) -> np.ndarray:
    return np.stack([
        np.r_[item["state"], item["action"], item["reward"], item["next_state"]]
        for item in transition_history(row, history_length)
    ])


def engineered_features(row: dict[str, Any], history_length: int,
                        action_penalty: float, goal_radius: float = 1.0) -> np.ndarray:
    history = transition_history(row, history_length)
    positions = np.stack([item["next_state"] for item in history])
    b = np.asarray([reward_equation_b(item["next_state"], item["action"], item["reward"],
                                     action_penalty, goal_radius) for item in history])
    gram = positions.T @ positions
    cross = positions.T @ b
    return np.r_[cross, gram[0, 0], gram[0, 1], gram[1, 1]]


def state_only_features(row: dict[str, Any]) -> np.ndarray:
    return np.asarray(row["states"][-1], dtype=float).copy()


def fit_ridge(rows: list[dict[str, Any]], feature: Callable[[dict[str, Any]], np.ndarray],
              alpha: float) -> np.ndarray:
    x = np.stack([np.r_[1.0, feature(row)] for row in rows])
    y = np.stack([[row["goal_cos"], row["goal_sin"]] for row in rows])
    penalty = np.eye(x.shape[1]) * float(alpha)
    penalty[0, 0] = 0.0
    return np.linalg.pinv(x.T @ x + penalty) @ x.T @ y


def predict_ridge(weights: np.ndarray, feature: np.ndarray) -> np.ndarray:
    return normalize_prediction(np.r_[1.0, np.asarray(feature, dtype=float)] @ weights)


def validate_transition_alignment(row: dict[str, Any], action_penalty: float,
                                  goal_radius: float = 1.0, tolerance: float = 2.0e-6) -> dict[str, Any]:
    states = np.asarray(row["states"], dtype=float)
    actions = np.clip(np.asarray(row["actions"], dtype=float), -1.0, 1.0)
    rewards = np.asarray(row["rewards"], dtype=float)
    goal = goal_radius * np.array([row["goal_cos"], row["goal_sin"]], dtype=float)
    expected = -(np.sum((states[1:] - goal) ** 2, axis=1) + action_penalty * np.sum(actions**2, axis=1))
    residual = np.abs(expected - rewards)
    if float(residual.max(initial=0.0)) > tolerance:
        raise ValueError("Reward is not aligned with the resulting position s_{t+1}")
    return {"transition_count": len(actions), "max_reward_residual": float(residual.max(initial=0.0))}


def validate_no_target_leakage(feature_fields: set[str] | frozenset[str]) -> None:
    leaked = FORBIDDEN_INPUT_FIELDS.intersection(feature_fields)
    if leaked:
        raise ValueError(f"Target leakage in probe inputs: {sorted(leaked)}")


def validate_split_isolation(fit_rows: list[dict[str, Any]]) -> None:
    found = {str(row["split"]) for row in fit_rows}
    if found != {"train"}:
        raise ValueError(f"Probe fitting must use train only, found {sorted(found)}")


def action_sequence(policy: str, horizon: int, seed: int) -> np.ndarray:
    if policy == "existing_exploratory":
        return np.random.default_rng(seed).uniform(-1.0, 1.0, size=(horizon, 2))
    if policy == "isotropic_random":
        return np.random.default_rng(seed).uniform(-1.0, 1.0, size=(horizon, 2))
    if policy == "deterministic_orthogonal":
        cycle = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        offset = (2 - horizon) % 4
        return np.stack([cycle[(offset + index) % 4] for index in range(horizon)])
    raise ValueError(f"Unknown behavior policy: {policy}")
