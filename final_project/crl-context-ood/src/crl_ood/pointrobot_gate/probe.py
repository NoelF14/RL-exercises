"""Pre-encoder ridge probes for reward-history identifiability."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from crl_ood.pointrobot_gate.spec import OOD_SPLITS, PRIMARY_SPLITS, circular_distance, context_splits

PROBE_FILES = (
    "probe_results_by_seed.csv", "probe_history_length_summary.csv",
    "probe_predictions.csv", "probe_error_vs_history.png",
)


def collect_probe_dataset(config: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    """Collect trajectories with paired actions that are identical across contexts."""
    from crl_ood.pointrobot_gate.environment import DenseSemiCirclePointRobot
    splits = context_splits(config)
    count = int(config["probe"]["trajectories_per_context"])
    offset = int(config["probe"]["action_seed_offset"])
    kwargs = {key: config["environment"][key] for key in (
        "goal_radius", "start_position", "reset_noise", "step_scale", "position_limit",
        "horizon", "action_penalty", "success_threshold",
    )}
    rows = []
    for trajectory_index in range(count):
        action_seed = offset + seed * 100_000 + trajectory_index
        actions = np.random.default_rng(action_seed).uniform(-1.0, 1.0, size=(kwargs["horizon"], 2))
        for split, goals in splits.items():
            for context_id, angle in goals.items():
                env = DenseSemiCirclePointRobot(angle, "hidden", **kwargs)
                state, _ = env.reset(seed=action_seed)
                states = [state.astype(float).tolist()]
                rewards = []
                for action in actions:
                    state, reward, _, truncated, _ = env.step(action)
                    states.append(state.astype(float).tolist()); rewards.append(float(reward))
                assert truncated
                rows.append({"seed": seed, "split": split, "context_id": context_id,
                             "goal_angle": angle, "goal_cos": math.cos(angle), "goal_sin": math.sin(angle),
                             "trajectory_index": trajectory_index, "action_seed": action_seed,
                             "states": np.asarray(states, dtype=float), "actions": actions.copy(),
                             "rewards": np.asarray(rewards, dtype=float)})
                env.close()
    return rows


def state_only_features(trajectory: dict[str, Any]) -> np.ndarray:
    """Current-state baseline; contains no action, reward, or history."""
    return np.asarray(trajectory["states"][-1], dtype=float).copy()


def history_features(trajectory: dict[str, Any], history_length: int) -> np.ndarray:
    """Flatten exactly H transitions as states, actions, rewards (no labels)."""
    if history_length <= 0 or history_length > len(trajectory["actions"]):
        raise ValueError("Invalid history length")
    states = np.asarray(trajectory["states"][-(history_length + 1):], dtype=float).reshape(-1)
    actions = np.asarray(trajectory["actions"][-history_length:], dtype=float).reshape(-1)
    rewards = np.asarray(trajectory["rewards"][-history_length:], dtype=float).reshape(-1)
    return np.concatenate((states, actions, rewards))


def run_probe(config_path: str | Path, output_dir: str | Path | None = None) -> dict[str, Path]:
    config_path = Path(config_path)
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    output = Path(output_dir) if output_dir else Path(config["experiment"]["results_dir"]) / "probe"
    output.mkdir(parents=True, exist_ok=True)
    all_predictions, seed_results = [], []
    histories = [int(x) for x in config["probe"]["history_lengths"]]
    alpha = float(config["probe"]["ridge_alpha"])
    for seed in map(int, config["probe"]["seeds"]):
        dataset = collect_probe_dataset(config, seed)
        for label, history in [("state_only", 0), *[(f"history_h{h}", h) for h in histories]]:
            feature = state_only_features if history == 0 else lambda row, h=history: history_features(row, h)
            train = [row for row in dataset if row["split"] == "train"]
            weights_cos, weights_sin = _fit_ridge(train, feature, alpha)
            for split in (*PRIMARY_SPLITS[1:], *OOD_SPLITS):
                test = [row for row in dataset if row["split"] == split]
                predictions = []
                for row in test:
                    x = np.r_[1.0, feature(row)]
                    pred_cos, pred_sin = float(x @ weights_cos), float(x @ weights_sin)
                    pred_angle = math.atan2(pred_sin, pred_cos)
                    error = circular_distance(pred_angle, row["goal_angle"])
                    prediction = {"seed": seed, "probe": label, "history_length": history,
                                  "split": split, "context_id": row["context_id"],
                                  "trajectory_index": row["trajectory_index"], "action_seed": row["action_seed"],
                                  "goal_angle": row["goal_angle"], "goal_cos": row["goal_cos"], "goal_sin": row["goal_sin"],
                                  "predicted_cos": pred_cos, "predicted_sin": pred_sin,
                                  "predicted_angle": pred_angle, "circular_angle_error": error}
                    all_predictions.append(prediction); predictions.append(prediction)
                seed_results.append({"seed": seed, "probe": label, "history_length": history, "split": split,
                                     "circular_angle_mae": statistics.fmean(x["circular_angle_error"] for x in predictions),
                                     "cos_mae": statistics.fmean(abs(x["predicted_cos"] - x["goal_cos"]) for x in predictions),
                                     "sin_mae": statistics.fmean(abs(x["predicted_sin"] - x["goal_sin"]) for x in predictions),
                                     "analysis_role": "gate" if split == "id" else "descriptive_only"})
    summary = _summary(seed_results)
    paths = {name: output / name for name in PROBE_FILES}
    _csv(paths[PROBE_FILES[0]], seed_results); _csv(paths[PROBE_FILES[1]], summary); _csv(paths[PROBE_FILES[2]], all_predictions)
    _plot(summary, paths[PROBE_FILES[3]])
    return paths


def _fit_ridge(rows: list[dict[str, Any]], feature: Any, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack([np.r_[1.0, feature(row)] for row in rows])
    penalty = np.eye(x.shape[1]) * alpha; penalty[0, 0] = 0.0
    solve = np.linalg.pinv(x.T @ x + penalty) @ x.T
    return solve @ np.asarray([row["goal_cos"] for row in rows]), solve @ np.asarray([row["goal_sin"] for row in rows])


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    keys = sorted({(row["probe"], row["history_length"], row["split"]) for row in rows})
    for probe, history, split in keys:
        values = [row["circular_angle_mae"] for row in rows if (row["probe"], row["history_length"], row["split"]) == (probe, history, split)]
        output.append({"probe": probe, "history_length": history, "split": split, "seed_count": len(values),
                       "mean_circular_angle_mae": statistics.fmean(values),
                       "std_across_seeds_descriptive": statistics.pstdev(values),
                       "confidence_interval": "not_computed_two_seeds",
                       "analysis_role": "gate" if split == "id" else "descriptive_only"})
    return output


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0])); writer.writeheader(); writer.writerows(rows)


def _plot(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    id_rows = sorted((row for row in rows if row["split"] == "id"), key=lambda row: int(row["history_length"]))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([row["history_length"] for row in id_rows], [row["mean_circular_angle_mae"] for row in id_rows], marker="o")
    ax.set(xlabel="History length (0 = state only)", ylabel="ID circular-angle MAE (rad)", title="Context identifiability from interaction history")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pointrobot_gate/gate.yaml"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    for path in run_probe(args.config, args.output_dir).values():
        print(path, flush=True)


if __name__ == "__main__":
    main()
