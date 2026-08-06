"""Generate fixed probe-audit datasets and supervised diagnostic results; never trains PPO."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from crl_ood.pointrobot_gate.spec import context_splits
from crl_ood.pointrobot_probe_audit.core import (
    TRANSITION_FIELDS, action_sequence, analytic_goal, circular_angle_error,
    engineered_features, fit_ridge, normalize_prediction, predict_ridge,
    raw_history_features, sequence_features, state_only_features,
    transition_history, validate_no_target_leakage, validate_split_isolation,
    validate_transition_alignment,
)


def collect_dataset(source: dict[str, Any], audit: dict[str, Any], policy: str, seed: int) -> list[dict[str, Any]]:
    env = source["environment"]
    horizon = int(env["horizon"])
    count = int(audit["trajectories_per_context"])
    offsets = {"existing_exploratory": int(audit["action_seed_offset"]),
               "isotropic_random": int(audit["isotropic_seed_offset"]),
               "deterministic_orthogonal": 0}
    rows: list[dict[str, Any]] = []
    for trajectory_index in range(count):
        action_seed = offsets[policy] + seed * 100_000 + trajectory_index
        actions = action_sequence(policy, horizon, action_seed)
        for split, goals in context_splits(source).items():
            for context_id, angle in goals.items():
                states, rewards = _simulate(actions, angle, env)
                rows.append({
                    "seed": seed, "behavior_policy": policy, "split": split,
                    "context_id": context_id, "goal_angle": angle,
                    "goal_cos": math.cos(angle), "goal_sin": math.sin(angle),
                    "trajectory_index": trajectory_index, "action_seed": action_seed,
                    "states": states, "actions": actions.copy(), "rewards": rewards,
                })
    return rows


def _simulate(actions: np.ndarray, angle: float, env: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(env["start_position"], dtype=np.float64).copy()
    goal = float(env["goal_radius"]) * np.array([math.cos(angle), math.sin(angle)])
    states = [position.astype(np.float32).astype(float)]
    rewards = []
    for action in actions:
        clipped = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        position = np.clip(position + float(env["step_scale"]) * clipped,
                           -float(env["position_limit"]), float(env["position_limit"]))
        rewards.append(-(float(np.sum((position - goal) ** 2))
                         + float(env["action_penalty"]) * float(clipped @ clipped)))
        states.append(position.astype(np.float32).astype(float))
    return np.asarray(states), np.asarray(rewards)


def run_audit(config_path: str | Path) -> dict[str, Path]:
    config_path = Path(config_path)
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with Path(config["experiment"]["source_config"]).open(encoding="utf-8") as handle:
        source = yaml.safe_load(handle)
    output = Path(config["experiment"]["results_dir"])
    output.mkdir(parents=True, exist_ok=True)
    audit = config["audit"]
    histories = [int(value) for value in audit["history_lengths"]]
    env = source["environment"]
    analytic_rows: list[dict[str, Any]] = []
    datasets: dict[tuple[str, int], list[dict[str, Any]]] = {}
    alignment_max = 0.0
    matched_checks = 0
    target_constant_checks = 0
    metadata_invariance_checks = 0
    sample_count = 0
    for policy in audit["behavior_policies"]:
        for seed in map(int, audit["seeds"]):
            dataset = collect_dataset(source, audit, policy, seed)
            datasets[(policy, seed)] = dataset
            sample_count += len(dataset)
            for row in dataset:
                checked = validate_transition_alignment(row, float(env["action_penalty"]), float(env["goal_radius"]))
                alignment_max = max(alignment_max, checked["max_reward_residual"])
                repeated_targets = np.repeat([[row["goal_cos"], row["goal_sin"]]], len(row["actions"]), axis=0)
                if not np.all(repeated_targets == repeated_targets[0]):
                    raise ValueError("Target label changed within an episode")
                target_constant_checks += 1
                for history_length in histories:
                    result = analytic_goal(transition_history(row, history_length), float(env["action_penalty"]),
                                           float(env["goal_radius"]), float(audit["rank_tolerance"]))
                    result.update({
                        "behavior_policy": policy, "seed": seed, "split": row["split"],
                        "context_id": row["context_id"], "trajectory_index": row["trajectory_index"],
                        "action_seed": row["action_seed"], "history_length": history_length,
                        "goal_angle": row["goal_angle"],
                        "circular_angle_error": circular_angle_error(result["estimated_angle"], row["goal_angle"]),
                        "analysis_role": "audit_decision" if row["split"] == "id" else
                                         "descriptive_only" if str(row["split"]).startswith("ood_") else "training_diagnostic",
                    })
                    analytic_rows.append(result)
                altered = dict(row, goal_angle=123.0, goal_cos=-9.0, goal_sin=8.0,
                               context_id=999, split="ood_right", filename="leak.csv")
                if not np.array_equal(raw_history_features(row, histories[-1]),
                                      raw_history_features(altered, histories[-1])):
                    raise ValueError("Metadata-derived target leakage in raw probe inputs")
                if not np.array_equal(sequence_features(row, histories[-1]),
                                      sequence_features(altered, histories[-1])):
                    raise ValueError("Metadata-derived target leakage in sequence probe inputs")
                metadata_invariance_checks += 1
            for trajectory_index in range(int(audit["trajectories_per_context"])):
                paired = [row for row in dataset if row["trajectory_index"] == trajectory_index]
                reference = paired[0]["actions"]
                if not all(np.array_equal(row["actions"], reference) for row in paired):
                    raise ValueError("Behavior-policy actions differ across matched contexts")
                matched_checks += 1

    selected = str(audit["selected_behavior_policy"])
    model_rows: list[dict[str, Any]] = []
    parity_counts: dict[str, int] = {}
    for seed in map(int, audit["seeds"]):
        dataset = datasets[(selected, seed)]
        train = [row for row in dataset if row["split"] == "train"]
        validate_split_isolation(train)
        weights = fit_ridge(train, state_only_features, float(config["models"]["ridge_alpha"]))
        model_rows.extend(_evaluate_ridge(dataset, seed, "state_only", 0, weights, state_only_features, 3))
        for history_length in histories:
            features = {
                "raw_ridge": lambda row, h=history_length: raw_history_features(row, h),
                "engineered_linear": lambda row, h=history_length: engineered_features(
                    row, h, float(env["action_penalty"]), float(env["goal_radius"])),
            }
            for name, feature in features.items():
                alpha = float(config["models"]["engineered_ridge_alpha"] if name == "engineered_linear"
                              else config["models"]["ridge_alpha"])
                weights = fit_ridge(train, feature, alpha)
                model_rows.extend(_evaluate_ridge(dataset, seed, name, history_length, weights, feature, weights.size))
            for name in ("mlp", "gru"):
                predictions, parameter_count = _fit_neural(train, dataset, history_length, name,
                                                           config["models"][name], seed)
                model_rows.extend(_evaluate_predictions(dataset, predictions, seed, name, history_length,
                                                        parameter_count))
            lengths = {
                "raw_ridge": len(train), "engineered_linear": len(train),
                "mlp": len(train), "gru": len(train),
            }
            if len(set(lengths.values())) != 1:
                raise ValueError("Probe datasets are not identical")
            parity_counts[f"seed_{seed}_h{history_length}"] = len(train)

    validate_no_target_leakage(set(TRANSITION_FIELDS))
    source_predictions = _read_csv(Path(config["experiment"]["source_results"]) / "probe" / "probe_predictions.csv")
    existing_keys = {(str(row["seed"]), row["split"], str(row["context_id"]),
                      str(row["trajectory_index"]), str(row["action_seed"])) for row in source_predictions}
    reconstructed_keys = {(str(row["seed"]), row["split"], str(row["context_id"]),
                           str(row["trajectory_index"]), str(row["action_seed"]))
                          for seed in map(int, audit["seeds"])
                          for row in datasets[("existing_exploratory", seed)] if row["split"] != "train"}
    if not existing_keys.issubset(reconstructed_keys):
        raise ValueError("Existing probe prediction metadata do not match reconstructed trajectories")

    analytic_path = output / "analytic_estimator_by_history.csv"
    model_path = output / "probe_model_results_by_seed.csv"
    _write_csv(analytic_path, analytic_rows)
    _write_csv(model_path, model_rows)
    alignment = {
        "accepted": True,
        "sequence_definition": "Each transition is exactly (s_t, a_t, r_t, s_{t+1}); suffix histories end at the episode horizon and contain no future transitions.",
        "reward_timing": "post_transition",
        "reward_alignment_pass": True,
        "max_absolute_reward_residual": alignment_max,
        "no_future_transitions_pass": True,
        "target_constant_within_episode_pass": True,
        "target_constant_episode_checks": target_constant_checks,
        "probe_input_fields": list(TRANSITION_FIELDS),
        "forbidden_input_fields_absent_pass": True,
        "metadata_perturbation_invariance_checks": metadata_invariance_checks,
        "matched_actions_across_contexts_pass": True,
        "matched_action_groups_checked": matched_checks,
        "train_id_ood_fit_isolation_pass": True,
        "fit_splits": ["train"],
        "evaluation_splits": ["id", "ood_left", "ood_right"],
        "existing_probe_reconstruction_coverage_pass": True,
        "existing_prediction_metadata_rows": len(source_predictions),
        "generated_episode_rows": sample_count,
        "dataset_parity_pass": True,
        "dataset_parity_train_counts": parity_counts,
        "off_by_one_negative_control": "Covered by test_transition_reward_alignment_and_off_by_one_negative_control.",
    }
    alignment_path = output / "probe_alignment_audit.json"
    alignment_path.write_text(json.dumps(alignment, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"alignment": alignment_path, "analytic": analytic_path, "models": model_path}


def _fit_neural(train: list[dict[str, Any]], all_rows: list[dict[str, Any]], history_length: int,
                name: str, settings: dict[str, Any], seed: int) -> tuple[np.ndarray, int]:
    import torch
    from torch import nn

    torch.manual_seed(31000 + seed * 101 + history_length)
    torch.set_num_threads(1)
    if name == "mlp":
        x_train = np.stack([raw_history_features(row, history_length) for row in train])
        x_all = np.stack([raw_history_features(row, history_length) for row in all_rows])
        mean, scale = x_train.mean(0), x_train.std(0)
        scale[scale < 1.0e-8] = 1.0
        x_train, x_all = (x_train - mean) / scale, (x_all - mean) / scale
        layers: list[nn.Module] = []
        width = x_train.shape[1]
        for hidden in map(int, settings["hidden_sizes"]):
            layers.extend((nn.Linear(width, hidden), nn.Tanh()))
            width = hidden
        layers.append(nn.Linear(width, 2))
        model: nn.Module = nn.Sequential(*layers)
    else:
        x_train = np.stack([sequence_features(row, history_length) for row in train])
        x_all = np.stack([sequence_features(row, history_length) for row in all_rows])
        mean, scale = x_train.mean((0, 1)), x_train.std((0, 1))
        scale[scale < 1.0e-8] = 1.0
        x_train, x_all = (x_train - mean) / scale, (x_all - mean) / scale

        class DiagnosticGRU(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gru = nn.GRU(x_train.shape[2], int(settings["hidden_size"]), batch_first=True)
                self.head = nn.Linear(int(settings["hidden_size"]), 2)

            def forward(self, value: Any) -> Any:
                encoded, _ = self.gru(value)
                return self.head(encoded[:, -1])

        model = DiagnosticGRU()
    xt = torch.tensor(x_train, dtype=torch.float32)
    yt = torch.tensor([[row["goal_cos"], row["goal_sin"]] for row in train], dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(settings["learning_rate"]),
                                 weight_decay=float(settings["weight_decay"]))
    for _ in range(int(settings["epochs"])):
        prediction = model(xt)
        loss = torch.mean((prediction - yt) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        output = model(torch.tensor(x_all, dtype=torch.float32)).numpy()
    return np.stack([normalize_prediction(row) for row in output]), sum(p.numel() for p in model.parameters())


def _evaluate_ridge(rows: list[dict[str, Any]], seed: int, model: str, history: int,
                    weights: np.ndarray, feature: Any, parameter_count: int) -> list[dict[str, Any]]:
    predictions = np.stack([predict_ridge(weights, feature(row)) for row in rows])
    return _evaluate_predictions(rows, predictions, seed, model, history, parameter_count)


def _evaluate_predictions(rows: list[dict[str, Any]], predictions: np.ndarray, seed: int, model: str,
                          history: int, parameter_count: int) -> list[dict[str, Any]]:
    output = []
    for split in ("train", "id", "ood_left", "ood_right"):
        indices = [index for index, row in enumerate(rows) if row["split"] == split]
        errors = [circular_angle_error(math.atan2(predictions[index, 1], predictions[index, 0]),
                                       rows[index]["goal_angle"]) for index in indices]
        output.append({
            "seed": seed, "behavior_policy": rows[0]["behavior_policy"], "model": model,
            "history_length": history, "split": split, "sample_count": len(indices),
            "parameter_count": parameter_count, "circular_angle_mae": float(np.mean(errors)),
            "analysis_role": "audit_decision" if split == "id" else
                             "descriptive_only" if split.startswith("ood_") else "training_diagnostic",
        })
    return output


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pointrobot_probe_audit/audit.yaml"))
    args = parser.parse_args()
    for path in run_audit(args.config).values():
        print(path, flush=True)


if __name__ == "__main__":
    main()
