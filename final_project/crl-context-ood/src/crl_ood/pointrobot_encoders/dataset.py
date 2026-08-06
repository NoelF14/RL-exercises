"""Deterministic shared trajectory data and leakage-safe history windows."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from crl_ood.pointrobot_gate.environment import DenseSemiCirclePointRobot
from crl_ood.utils.paths import project_root

TRAIN_CONTEXTS = (-0.6, -0.3, 0.0, 0.3, 0.6)
TRANSITION_FIELDS = ("state", "action", "reward", "next_state")
ENCODER_INPUT_FIELDS = TRANSITION_FIELDS


def load_spec(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("encoder configuration must be a mapping")
    if tuple(float(x) for x in value["dataset"]["train_contexts"]) != TRAIN_CONTEXTS:
        raise ValueError("encoder fitting contexts must be exactly the five PointRobot training contexts")
    return value


def behavior_actions(policy: str, horizon: int, trajectory_seed: int,
                     prefix: list[list[float]]) -> np.ndarray:
    """Context-independent action sequence shared by every matched goal."""
    rng = np.random.default_rng(int(trajectory_seed))
    random = rng.uniform(-1.0, 1.0, size=(horizon, 2)).astype(np.float32)
    if policy == "random_only":
        return random
    if policy != "orthogonal_then_isotropic":
        raise ValueError(f"unknown behavior policy {policy!r}")
    fixed = np.asarray(prefix, dtype=np.float32)
    if fixed.ndim != 2 or fixed.shape[1] != 2 or len(fixed) > horizon:
        raise ValueError("orthogonal prefix must be an Hx2 action array no longer than the episode")
    random[:len(fixed)] = fixed
    return random


def pointrobot_reward(next_state: np.ndarray, action: np.ndarray, goal_angle: float,
                      *, goal_radius: float = 1.0, action_penalty: float = 0.01) -> np.ndarray:
    goal = goal_radius * np.asarray([np.cos(goal_angle), np.sin(goal_angle)])
    return -(np.sum((np.asarray(next_state) - goal) ** 2, axis=-1)
             + action_penalty * np.sum(np.clip(action, -1.0, 1.0) ** 2, axis=-1))


def collect_arrays(config: dict[str, Any], budget: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    data = config["dataset"]
    if budget not in data["budgets"]:
        raise ValueError(f"unknown dataset budget {budget!r}")
    per_context = int(data["budgets"][budget]["trajectories_per_context"])
    horizon = int(data["horizon"])
    policy = str(data["behavior_policy"])
    offset = int(data["action_seed_offset"])
    with (project_root() / config["experiment"]["source_gate_config"]).open(encoding="utf-8") as handle:
        gate = yaml.safe_load(handle)
    env_cfg = gate["environment"]
    env_kwargs = {key: env_cfg[key] for key in (
        "goal_radius", "start_position", "reset_noise", "step_scale", "position_limit",
        "horizon", "action_penalty", "success_threshold")}
    if horizon != int(env_kwargs["horizon"]):
        raise ValueError("dataset and environment horizons differ")
    records: dict[str, list[Any]] = {key: [] for key in (
        "states", "actions", "rewards", "next_states", "terminated", "truncated",
        "contexts", "timesteps", "trajectory_seeds", "episode_ids", "assignments")}
    split_rng = np.random.default_rng(int(data["split_seed"]))
    validation_n = max(1, int(round(per_context * float(data["validation_fraction"]))))
    validation_indices = set(split_rng.choice(per_context, size=validation_n, replace=False).tolist())
    for context_index, angle in enumerate(TRAIN_CONTEXTS):
        for trajectory_index in range(per_context):
            trajectory_seed = offset + trajectory_index
            actions = behavior_actions(policy, horizon, trajectory_seed, data["orthogonal_prefix"])
            env = DenseSemiCirclePointRobot(angle, "hidden", **env_kwargs)
            state, _ = env.reset(seed=trajectory_seed)
            states, rewards, next_states, terminated, truncated = [state], [], [], [], []
            for action in actions:
                nxt, reward, term, trunc, _ = env.step(action)
                rewards.append(reward); next_states.append(nxt); terminated.append(term); truncated.append(trunc)
                states.append(nxt)
            env.close()
            episode_id = context_index * per_context + trajectory_index
            records["states"].append(states)
            records["actions"].append(actions)
            records["rewards"].append(rewards)
            records["next_states"].append(next_states)
            records["terminated"].append(terminated)
            records["truncated"].append(truncated)
            records["contexts"].append(angle)
            records["timesteps"].append(np.arange(horizon))
            records["trajectory_seeds"].append(trajectory_seed)
            records["episode_ids"].append(episode_id)
            records["assignments"].append(1 if trajectory_index in validation_indices else 0)
    arrays = {
        "states": np.asarray(records["states"], dtype=np.float32),
        "actions": np.asarray(records["actions"], dtype=np.float32),
        "rewards": np.asarray(records["rewards"], dtype=np.float32),
        "next_states": np.asarray(records["next_states"], dtype=np.float32),
        "terminated": np.asarray(records["terminated"], dtype=np.bool_),
        "truncated": np.asarray(records["truncated"], dtype=np.bool_),
        "contexts": np.asarray(records["contexts"], dtype=np.float32),
        "timesteps": np.asarray(records["timesteps"], dtype=np.int32),
        "trajectory_seeds": np.asarray(records["trajectory_seeds"], dtype=np.int64),
        "episode_ids": np.asarray(records["episode_ids"], dtype=np.int64),
        "assignments": np.asarray(records["assignments"], dtype=np.int8),
    }
    features = transition_features(arrays["states"][:, :-1], arrays["actions"], arrays["rewards"], arrays["next_states"])
    fit = features[arrays["assignments"] == 0].reshape(-1, features.shape[-1]).astype(np.float64)
    mean, std = fit.mean(axis=0), fit.std(axis=0)
    std[std < 1e-8] = 1.0
    arrays["normalization_mean"] = mean.astype(np.float32)
    arrays["normalization_std"] = std.astype(np.float32)
    metadata = {
        "format_version": 1, "budget": budget, "immutable": True,
        "source_git_commit": _commit(), "environment_spec": DenseSemiCirclePointRobot(0.0).environment_spec,
        "behavior_policy": {"name": policy, "orthogonal_prefix": data["orthogonal_prefix"],
                            "random_distribution": "iid Uniform([-1,1]^2) after prefix"},
        "contexts": list(TRAIN_CONTEXTS), "trajectory_seeds": sorted(set(arrays["trajectory_seeds"].tolist())),
        "assignment_encoding": {"0": "train", "1": "validation"},
        "episode_count": int(len(arrays["episode_ids"])), "transition_count": int(len(arrays["episode_ids"]) * horizon),
        "array_schema": {key: {"dtype": str(value.dtype), "shape": list(value.shape)} for key, value in arrays.items()},
        "normalization": {"fit_scope": "training assignment only, training contexts only",
                          "mean": arrays["normalization_mean"].tolist(), "std": arrays["normalization_std"].tolist()},
    }
    metadata["dataset_checksum"] = dataset_checksum(arrays, metadata)
    return arrays, metadata


def transition_features(state: np.ndarray, action: np.ndarray, reward: np.ndarray,
                        next_state: np.ndarray) -> np.ndarray:
    reward_column = np.asarray(reward)[..., None]
    return np.concatenate((np.asarray(state), np.asarray(action), reward_column, np.asarray(next_state)), axis=-1)


def dataset_checksum(arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode()); digest.update(str(value.dtype).encode()); digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    clean = {key: value for key, value in metadata.items() if key != "dataset_checksum"}
    digest.update(json.dumps(clean, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def save_dataset(output_dir: str | Path, arrays: dict[str, np.ndarray], metadata: dict[str, Any],
                 *, overwrite: bool = False) -> Path:
    path = Path(output_dir)
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"dataset directory is nonempty: {path}")
        raise FileExistsError("immutable datasets are never overwritten; choose a new deterministic namespace")
    path.mkdir(parents=True, exist_ok=True)
    np.savez(path / "trajectories.npz", **arrays)
    (path / "dataset.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (path / "dataset.sha256").write_text(metadata["dataset_checksum"] + "\n", encoding="ascii")
    return path


def load_dataset(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    root = Path(path)
    with np.load(root / "trajectories.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    metadata = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    actual = dataset_checksum(arrays, metadata)
    if actual != metadata["dataset_checksum"] or actual != (root / "dataset.sha256").read_text().strip():
        raise ValueError("dataset checksum mismatch")
    return arrays, metadata


@dataclass(frozen=True)
class WindowIndex:
    episode: int
    timestep: int


def window_indices(arrays: dict[str, np.ndarray], assignment: str, history_length: int,
                   future_horizon: int) -> list[WindowIndex]:
    code = {"train": 0, "validation": 1}[assignment]
    horizon = arrays["actions"].shape[1]
    # t is a decision time. History uses max(0,t-H)..t-1; future uses t..t+K-1.
    return [WindowIndex(e, t) for e in range(len(arrays["episode_ids"])) if arrays["assignments"][e] == code
            for t in range(0, horizon - future_horizon + 1)]


def make_window(arrays: dict[str, np.ndarray], index: WindowIndex, history_length: int,
                future_horizon: int) -> dict[str, np.ndarray]:
    e, t = index.episode, index.timestep
    start = max(0, t - history_length)
    raw = transition_features(arrays["states"][e, start:t], arrays["actions"][e, start:t],
                              arrays["rewards"][e, start:t], arrays["next_states"][e, start:t])
    length = len(raw)
    history = np.zeros((history_length, 7), dtype=np.float32)
    mask = np.zeros(history_length, dtype=np.bool_)
    if length:
        history[:length] = (raw - arrays["normalization_mean"]) / arrays["normalization_std"]
        mask[:length] = True
    future_states = arrays["states"][e, t:t + future_horizon + 1]
    return {
        "history": history, "mask": mask, "length": np.asarray(length, dtype=np.int64),
        "current_state": arrays["states"][e, t].copy(),
        "future_actions": arrays["actions"][e, t:t + future_horizon].copy(),
        "future_state_deltas": np.diff(future_states, axis=0).copy(),
        "future_rewards": arrays["rewards"][e, t:t + future_horizon].copy(),
        "future_states": arrays["next_states"][e, t:t + future_horizon].copy(),
        "context": arrays["contexts"][e].copy(), "episode_id": arrays["episode_ids"][e].copy(),
        "timestep": np.asarray(t, dtype=np.int64),
    }


def _commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_root(), check=True,
                          capture_output=True, text=True).stdout.strip()
