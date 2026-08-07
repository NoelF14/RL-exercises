"""Frozen Torch/environment evaluation. This module is never imported by result analysis."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml

from crl_ood.pointrobot_encoders.dataset import load_dataset, transition_features
from crl_ood.pointrobot_encoders.training import load_frozen_checkpoint
from crl_ood.pointrobot_gate.environment import DenseSemiCirclePointRobot

from .manifest import write_manifest
from .spec import METHODS, SEEDS, SPLITS, build_checkpoint_jobs, validate_primary_provenance


def _csv(path: Path, rows: list[dict[str, Any]], columns: Iterable[str] | None = None) -> None:
    fields = list(columns or (rows[0] if rows else ()))
    if not fields:
        raise ValueError(f"cannot write schema-free CSV {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def circular_absolute_error(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    delta = np.asarray(prediction, dtype=float) - np.asarray(target, dtype=float)
    return np.abs(np.arctan2(np.sin(delta), np.cos(delta)))


def fit_linear_train_only(features: np.ndarray, targets: np.ndarray, splits: np.ndarray) -> np.ndarray:
    mask = np.asarray(splits) == "train"
    if not mask.any() or set(np.asarray(splits)[mask]) != {"train"}:
        raise ValueError("linear probe fit data must be train-context samples only")
    design = np.column_stack((np.ones(mask.sum()), np.asarray(features, dtype=float)[mask]))
    return np.linalg.lstsq(design, np.asarray(targets, dtype=float)[mask], rcond=None)[0]


def predict_linear(coefficients: np.ndarray, features: np.ndarray) -> np.ndarray:
    return np.column_stack((np.ones(len(features)), np.asarray(features, dtype=float))) @ coefficients


def fit_pca_train_only(latents: np.ndarray, splits: np.ndarray, components: int = 2) -> dict[str, np.ndarray]:
    mask = np.asarray(splits) == "train"
    train = np.asarray(latents, dtype=float)[mask]
    if not len(train) or set(np.asarray(splits)[mask]) != {"train"}:
        raise ValueError("PCA fit data must be train-context samples only")
    mean = train.mean(axis=0)
    _, singular, vt = np.linalg.svd(train - mean, full_matrices=False)
    explained = singular**2 / max(1, len(train) - 1)
    return {"mean": mean, "components": vt[:components], "explained_variance": explained[:components],
        "explained_variance_ratio": explained[:components] / explained.sum()}


def _metric_rows(rows: list[dict[str, Any]], prediction_key: str, error_key: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    angles, splits = [], []
    for split, contexts in SPLITS.items():
        selected_split = [row for row in rows if row["split"] == split]
        splits.append({"split": split, "circular_angle_mae": float(np.mean([row[error_key] for row in selected_split])),
            "sample_count": len(selected_split), "fit_split": "train", "prediction_field": prediction_key})
        for angle in contexts:
            selected = [row for row in selected_split if np.isclose(row["goal_angle"], angle)]
            distance = "near" if split.startswith("ood") and np.isclose(abs(angle), .8) else (
                "far" if split.startswith("ood") and np.isclose(abs(angle), 1.0) else "not_ood")
            angles.append({"split": split, "goal_angle": angle, "ood_distance_group": distance,
                "circular_angle_mae": float(np.mean([row[error_key] for row in selected])),
                "sample_count": len(selected), "fit_split": "train"})
    return angles, splits


def _diagnostic_trajectories(spec: dict[str, Any], root: Path, output: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    arrays, metadata = load_dataset(root / spec["dataset"]["path"])
    n = int(spec["diagnostic_trajectories"]["trajectories_per_context"])
    reference_actions = arrays["actions"][:n].copy()
    reference_seeds = arrays["trajectory_seeds"][:n].astype(int)
    expected = np.asarray(spec["diagnostic_trajectories"]["trajectory_seeds"], dtype=int)
    if not np.array_equal(reference_seeds, expected):
        raise ValueError("immutable dataset does not contain the prespecified diagnostic trajectory seeds")
    with (root / "configs/pointrobot_gate/gate.yaml").open(encoding="utf-8") as handle:
        env_cfg = yaml.safe_load(handle)["environment"]
    kwargs = {key: env_cfg[key] for key in ("goal_radius", "start_position", "reset_noise", "step_scale",
        "position_limit", "horizon", "action_penalty", "success_threshold")}
    records, states, rewards = [], [], []
    trajectory_id = 0
    for split, contexts in SPLITS.items():
        for angle in contexts:
            for index, actions in enumerate(reference_actions):
                env = DenseSemiCirclePointRobot(angle, "hidden", **kwargs)
                state, _ = env.reset(seed=int(reference_seeds[index])); episode_states, episode_rewards = [state], []
                for action in actions:
                    state, reward, _, _, _ = env.step(action)
                    episode_states.append(state); episode_rewards.append(reward)
                env.close()
                states.append(episode_states); rewards.append(episode_rewards)
                records.append({"split": split, "goal_angle": angle, "trajectory_id": trajectory_id,
                    "trajectory_seed": int(reference_seeds[index]), "action_sequence_index": index})
                trajectory_id += 1
    trajectory_dir = output / "evaluation_protocol"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    archive = trajectory_dir / "diagnostic_trajectories.npz"
    np.savez_compressed(archive, actions=reference_actions, trajectory_seeds=reference_seeds,
        states=np.asarray(states, dtype=np.float32), rewards=np.asarray(rewards, dtype=np.float32))
    index_path = trajectory_dir / "diagnostic_trajectory_index.csv"; _csv(index_path, records)
    protocol_path = trajectory_dir / "provenance.json"
    protocol_path.write_text(json.dumps({"dataset_checksum": metadata["dataset_checksum"],
        "behavior_policy": spec["diagnostic_trajectories"]["behavior_policy"],
        "action_source": spec["diagnostic_trajectories"]["action_source"],
        "trajectory_seeds": reference_seeds.tolist(), "horizon": int(reference_actions.shape[1]),
        "action_sequences_shared_across_methods_seeds_and_contexts": True,
        "ppo_trajectories_used": False, "configuration_checksum": spec["_configuration_checksum"]},
        indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return records, [archive, index_path, protocol_path]


def _evaluate_job(spec: dict[str, Any], job: Any, checkpoint_row: dict[str, Any], root: Path,
                  trajectory_records: list[dict[str, Any]], protocol_archive: Path) -> list[Path]:
    output = job.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"representation evaluation directory is nonempty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with np.load(protocol_archive, allow_pickle=False) as archive:
        actions, states, rewards = archive["actions"], archive["states"], archive["rewards"]
    model, payload = load_frozen_checkpoint(job.checkpoint, expected_method=job.method,
        expected_dataset_checksum=spec["dataset"]["checksum"])
    normalization = payload["normalization"]
    mean, std = np.asarray(normalization["mean"], dtype=np.float32), np.asarray(normalization["std"], dtype=np.float32)
    sample_rows, latent_values, current_states = [], [], []
    history_length = int(spec["history"]["length"]); latent_dim = int(spec["latent"]["dimension"])
    model.eval()
    with torch.no_grad():
        for record, episode_states, episode_rewards in zip(trajectory_records, states, rewards, strict=True):
            action_sequence = actions[int(record["action_sequence_index"])]
            raw = transition_features(episode_states[:-1], action_sequence, episode_rewards, episode_states[1:])
            normalized = (raw - mean) / std
            for timestep in range(len(action_sequence)):
                start = max(0, timestep - history_length); length = timestep - start
                history = np.zeros((1, history_length, 7), dtype=np.float32)
                if length:
                    history[0, :length] = normalized[start:timestep]
                lengths = torch.as_tensor([length], dtype=torch.long)
                mask = torch.arange(history_length)[None, :] < lengths[:, None]
                latent = model.encode(torch.from_numpy(history), lengths, mask, deterministic=True)[0].cpu().numpy()
                if latent.shape != (latent_dim,):
                    raise ValueError("latent dimensionality is not 8")
                if length == 0 and not np.array_equal(latent, np.zeros(latent_dim, dtype=latent.dtype)):
                    raise ValueError("trained encoder empty-history mapping is not exact zero")
                base = {"method": job.method, "encoder_seed": job.encoder_seed, "split": record["split"],
                    "goal_angle": record["goal_angle"], "trajectory_id": record["trajectory_id"],
                    "trajectory_seed": record["trajectory_seed"], "timestep": timestep, "history_length": length,
                    "dataset_checksum": spec["dataset"]["checksum"], "checkpoint_path": str(job.checkpoint.resolve()),
                    "checkpoint_sha256": job.checkpoint_sha256,
                    "source_commit": spec["experiment"]["authoritative_primary_source_snapshot"],
                    "configuration_checksum": spec["_configuration_checksum"]}
                base.update({f"z_{index}": float(value) for index, value in enumerate(latent)})
                sample_rows.append(base); latent_values.append(latent); current_states.append(episode_states[timestep])
    z = np.asarray(latent_values); state = np.asarray(current_states); target = np.asarray([r["goal_angle"] for r in sample_rows])
    split_values = np.asarray([r["split"] for r in sample_rows])
    coefficients = fit_linear_train_only(z, target, split_values); predictions = predict_linear(coefficients, z)
    errors = circular_absolute_error(predictions, target)
    state_coefficients = fit_linear_train_only(state, target, split_values); state_predictions = predict_linear(state_coefficients, state)
    state_errors = circular_absolute_error(state_predictions, target)
    prediction_rows, state_rows = [], []
    for index, row in enumerate(sample_rows):
        identity = {key: row[key] for key in ("method", "encoder_seed", "split", "goal_angle", "trajectory_id",
            "trajectory_seed", "timestep", "history_length")}
        prediction_rows.append({**identity, "predicted_goal_angle": float(predictions[index]),
            "circular_absolute_angle_error": float(errors[index]), "probe_fit_split": "train"})
        state_rows.append({**identity, "current_state_x": float(state[index, 0]), "current_state_y": float(state[index, 1]),
            "predicted_goal_angle": float(state_predictions[index]),
            "circular_absolute_angle_error": float(state_errors[index]), "probe_fit_split": "train",
            "features": "current_state_only", "contains_history": False})
    angle_rows, seed_rows = _metric_rows(prediction_rows, "predicted_goal_angle", "circular_absolute_angle_error")
    for row in angle_rows + seed_rows:
        row.update({"method": job.method, "encoder_seed": job.encoder_seed})
    pca = fit_pca_train_only(z, split_values, int(spec["pca"]["components"]))
    coordinates = (z - pca["mean"]) @ pca["components"].T
    pca_rows = [{"method": job.method, "encoder_seed": job.encoder_seed, "split": row["split"],
        "goal_angle": row["goal_angle"], "trajectory_id": row["trajectory_id"], "trajectory_seed": row["trajectory_seed"],
        "timestep": row["timestep"], "pc_1": float(coordinates[index, 0]), "pc_2": float(coordinates[index, 1]),
        "pca_fit_split": "train", "standardization": "none", "cross_seed_alignment": False}
        for index, row in enumerate(sample_rows)]
    paths = {
        "latent_samples.csv": sample_rows, "probe_predictions.csv": prediction_rows,
        "probe_by_angle.csv": angle_rows, "probe_by_seed.csv": seed_rows,
        "state_only_probe.csv": state_rows, "pca_coordinates.csv": pca_rows,
    }
    written = []
    for name, rows in paths.items():
        path = output / name; _csv(path, rows); written.append(path)
    pca_path = output / "pca_model.npz"
    np.savez(pca_path, **pca, fit_split=np.asarray("train"), standardization=np.asarray("none")); written.append(pca_path)
    probe_path = output / "probe_model.npz"
    np.savez(probe_path, coefficients=coefficients, state_only_coefficients=state_coefficients,
        fit_split=np.asarray("train")); written.append(probe_path)
    provenance_path = output / "provenance.json"
    provenance_path.write_text(json.dumps({**checkpoint_row, "history_convention": spec["history"]["convention"],
        "diagnostic_trajectory_protocol": str(protocol_archive.resolve()), "probe_fit_split": "train",
        "pca_fit_split": "train", "pca_standardization": "none", "cross_seed_alignment": False,
        "selection_role": "diagnostic_only"}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    written.append(provenance_path)
    return written


def evaluate_all(spec: dict[str, Any], root: str | Path) -> Path:
    base = Path(root).resolve(); output = base / spec["experiment"]["results_dir"]
    checkpoint_rows = validate_primary_provenance(spec, base)
    jobs = build_checkpoint_jobs(spec, base)
    trajectory_records, files = _diagnostic_trajectories(spec, base, output)
    protocol_archive = files[0]
    lookup = {(row["method"], row["encoder_seed"]): row for row in checkpoint_rows}
    for job in jobs:
        files.extend(_evaluate_job(spec, job, lookup[(job.method, job.encoder_seed)], base,
            trajectory_records, protocol_archive))
    manifest = output / spec["analysis"]["evaluation_manifest"]
    write_manifest(output, files, manifest)
    return manifest
