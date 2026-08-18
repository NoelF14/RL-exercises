"""Deterministic matched encoder training, validation, and checkpoint handling."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from crl_ood.pointrobot_encoders.dataset import (TRAIN_CONTEXTS, WindowIndex, load_dataset,
                                                  make_window, pointrobot_reward, transition_features,
                                                  window_indices)
from crl_ood.pointrobot_encoders.models import (ContrastiveHistoryEncoder, build_model,
                                                 checkpoint_payload, contrastive_objective,
                                                 parameter_counts, vae_objective)
from crl_ood.utils.paths import project_root
from crl_ood.pointrobot_gate.environment import DenseSemiCirclePointRobot
from crl_ood.pointrobot_gate.spec import EXPECTED_SPLITS


class WindowDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, arrays: dict[str, np.ndarray], assignment: str, history: int, future: int) -> None:
        self.arrays, self.history, self.future = arrays, history, future
        self.indices = window_indices(arrays, assignment, history, future)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        row = make_window(self.arrays, self.indices[item], self.history, self.future)
        return {key: torch.as_tensor(value) for key, value in row.items()}


def hard_negative_rewards(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Relabel identical future state/actions under a deterministic different train goal."""
    contexts = batch["context"].detach().cpu().numpy()
    states = batch["future_states"].detach().cpu().numpy()
    actions = batch["future_actions"].detach().cpu().numpy()
    output, provenance = [], []
    for index, (context, future_states, future_actions) in enumerate(zip(contexts, states, actions)):
        source = min(range(len(TRAIN_CONTEXTS)), key=lambda i: abs(TRAIN_CONTEXTS[i] - float(context)))
        negative_goal = TRAIN_CONTEXTS[(source + 1) % len(TRAIN_CONTEXTS)]
        rewards = pointrobot_reward(future_states, future_actions, negative_goal).astype(np.float32)
        different = bool(not np.allclose(
            rewards, batch["future_rewards"][index].detach().cpu().numpy()))
        if not different:
            raise ValueError("hard negative must have a different reward target")
        output.append(rewards)
        provenance.append({"batch_index": index, "positive_goal": float(TRAIN_CONTEXTS[source]),
                           "negative_goal": float(negative_goal), "state_action_preserved": True,
                           "reward_targets_different": different})
    tensor = torch.as_tensor(np.asarray(output), device=batch["future_rewards"].device)
    return tensor, provenance

def hard_negative_rewards_alternative(
    batch: dict[str, torch.Tensor],
    rng: random.Random,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Relabel identical future state/actions under a sampled adjacent training goal."""
    contexts = batch["context"].detach().cpu().numpy()
    states = batch["future_states"].detach().cpu().numpy()
    actions = batch["future_actions"].detach().cpu().numpy()
    output, provenance = [], []
    for index, (context, future_states, future_actions) in enumerate(zip(contexts, states, actions)):
        source = min(range(len(TRAIN_CONTEXTS)), key=lambda i: abs(TRAIN_CONTEXTS[i] - float(context)))
        # Sample uniformly from available adjacent goals; endpoints have one candidate.
        candidates = []
        if source > 0:
            candidates.append(source - 1)
        if source < len(TRAIN_CONTEXTS) - 1:
            candidates.append(source + 1)

        negative_source = rng.choice(candidates)
        negative_goal = TRAIN_CONTEXTS[negative_source]

        rewards = pointrobot_reward(future_states, future_actions, negative_goal).astype(np.float32)
        different = bool(not np.allclose(
            rewards, batch["future_rewards"][index].detach().cpu().numpy()))
        if not different:
            raise ValueError("hard negative must have a different reward target")
        output.append(rewards)
        provenance.append({"batch_index": index, "positive_goal": float(TRAIN_CONTEXTS[source]),
                           "negative_goal": float(negative_goal), "state_action_preserved": True,
                           "reward_targets_different": different})
    tensor = torch.as_tensor(np.asarray(output), device=batch["future_rewards"].device)
    return tensor, provenance


def train_encoder(config: dict[str, Any], dataset_dir: str | Path, method: str, seed: int,
                  output_dir: str | Path, *, max_updates: int | None = None,
                  overwrite: bool = False, resume: bool = False) -> Path:
    arrays, metadata = load_dataset(dataset_dir)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()) and not resume:
        if overwrite:
            raise FileExistsError("atomic encoder runs are never overwritten; use a new directory")
        raise FileExistsError(f"run directory is nonempty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    encoder = config["encoder"]
    updates = int(max_updates if max_updates is not None else encoder["max_updates"])
    if updates <= 0:
        raise ValueError("max_updates must be positive")
    _seed(seed, bool(config["reproducibility"]["deterministic_torch"]))
    negative_rng = random.Random(seed)
    device = torch.device(str(config["reproducibility"]["device"]))
    model = build_model(method, encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(encoder["learning_rate"]))
    start_update, best_loss = 0, float("inf")
    final_path = output / "final.pt"
    if resume:
        if not final_path.is_file():
            raise FileNotFoundError("resume requires an existing final.pt")
        saved = torch.load(final_path, map_location=device, weights_only=False)
        if saved["dataset_checksum"] != metadata["dataset_checksum"] or saved["method"] != method:
            raise ValueError("resume checkpoint provenance mismatch")
        model.load_state_dict(saved["state_dict"])
        if "optimizer_state_dict" in saved:
            optimizer.load_state_dict(saved["optimizer_state_dict"])
        start_update = int(saved["update"])
        best_loss = float(saved.get("best_validation_loss", saved["validation_loss"]))
    history = int(encoder["history_length"]); future = int(encoder["future_horizon"])
    train_data = WindowDataset(arrays, "train", history, future)
    validation_data = WindowDataset(arrays, "validation", history, future)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_data, batch_size=int(encoder["batch_size"]), shuffle=True,
                        generator=generator, drop_last=False)
    validation_interval = min(int(encoder["validation_interval"]), updates)
    rows: list[dict[str, Any]] = []
    negative_rows: list[dict[str, Any]] = []
    iterator = iter(loader)
    for update in range(start_update + 1, updates + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader); batch = next(iterator)
        batch = {key: value.to(device) for key, value in batch.items()}
        model.train(); optimizer.zero_grad(set_to_none=True)
        losses, provenance = _loss(
            model,
            method,
            batch,
            config,
            rng=negative_rng,
        )
        losses["total"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(encoder["gradient_clip_norm"]))
        optimizer.step()
        if provenance and not negative_rows:
            negative_rows = [{**row, "episode_id": int(batch["episode_id"][i]),
                              "timestep": int(batch["timestep"][i])} for i, row in enumerate(provenance)]
        should_validate = update == updates or update % validation_interval == 0
        if should_validate:
            validation = validate(
                model,
                method,
                validation_data,
                config,
                device,
                negative_seed=seed,
            )
            row = {"update": update, "split": "validation", "learning_rate": optimizer.param_groups[0]["lr"],
                   "gradient_norm": float(gradient_norm), **validation}
            rows.append(row)
            payload = checkpoint_payload(model, method, config, metadata["normalization"],
                                         metadata["dataset_checksum"], seed, update, validation["total"])
            payload["optimizer_state_dict"] = optimizer.state_dict()
            payload["best_validation_loss"] = min(best_loss, validation["total"])
            torch.save(payload, final_path)
            if validation["total"] < best_loss:
                best_loss = validation["total"]
                torch.save(payload, output / "best.pt")
    _write_run_artifacts(output, config, metadata, method, seed, model, rows, negative_rows)
    return output


def _loss(
    model: torch.nn.Module,
    method: str,
    batch: dict[str, torch.Tensor],
    config: dict[str, Any],
    *,
    rng: random.Random | None = None,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    if method == "vae":
        losses = vae_objective(model(batch), batch, float(config["vae"]["state_loss_weight"]),
                               float(config["vae"]["reward_loss_weight"]), float(config["vae"]["kl_weight"]))
        return losses, []
    if method == "contrastive":
        negative, provenance = hard_negative_rewards(batch)
        losses = contrastive_objective(model, batch, negative, float(config["contrastive"]["temperature"]),
                                           str(config["contrastive"]["negative_mode"]))
    if method == "contrastive_alternative":
        if rng is None:
            raise ValueError(
                "contrastive_alternative requires an explicit RNG"
            )
        negative, provenance = hard_negative_rewards_alternative(batch, rng)
        losses = contrastive_objective(model, batch, negative, float(config["contrastive_alternative"]["temperature"]),
                                           str(config["contrastive_alternative"]["negative_mode"]))
    return losses, provenance


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    method: str,
    dataset: WindowDataset,
    config: dict[str, Any],
    device: torch.device,
    *,
    negative_seed: int = 0,
) -> dict[str, float]:
    model.eval(); totals: dict[str, list[float]] = {}
    negative_rng = random.Random(negative_seed)
    loader = DataLoader(dataset, batch_size=int(config["encoder"]["batch_size"]), shuffle=False)
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        losses, _ = _loss(
            model,
            method,
            batch,
            config,
            rng=negative_rng,
        )
        for key, value in losses.items():
            if key != "logits":
                totals.setdefault(key, []).append(float(value))
    return {key: float(np.mean(values)) for key, values in totals.items()}


def load_frozen_checkpoint(path: str | Path, expected_method: str | None = None,
                           expected_dataset_checksum: str | None = None) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if expected_method is not None and payload["method"] != expected_method:
        raise ValueError("encoder checkpoint method mismatch")
    if expected_dataset_checksum is not None and payload["dataset_checksum"] != expected_dataset_checksum:
        raise ValueError("encoder checkpoint dataset mismatch")
    model = build_model(payload["method"], payload["config"]["encoder"])
    model.load_state_dict(payload["state_dict"]); model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    payload["checkpoint_checksum"] = file_sha256(path)
    return model, payload


def evaluate_frozen(dataset_dir: str | Path, checkpoint: str | Path, output_dir: str | Path) -> Path:
    arrays, metadata = load_dataset(dataset_dir)
    model, payload = load_frozen_checkpoint(checkpoint, expected_dataset_checksum=metadata["dataset_checksum"])
    encoder = payload["config"]["encoder"]
    history_length, future_horizon = int(encoder["history_length"]), int(encoder["future_horizon"])
    rows, latents, reward_errors = [], [], []
    gate_path = project_root() / payload["config"]["experiment"]["source_gate_config"]
    with gate_path.open(encoding="utf-8") as handle:
        gate = yaml.safe_load(handle)
    env_cfg = gate["environment"]
    kwargs = {key: env_cfg[key] for key in ("goal_radius", "start_position", "reset_noise", "step_scale",
              "position_limit", "horizon", "action_penalty", "success_threshold")}
    per_context = len(arrays["episode_ids"]) // len(TRAIN_CONTEXTS)
    reference_actions = arrays["actions"][:per_context]
    reference_seeds = arrays["trajectory_seeds"][:per_context]
    reference_assignments = arrays["assignments"][:per_context]
    model.eval()
    with torch.no_grad():
        episode_id = 0
        for split, contexts in EXPECTED_SPLITS.items():
            for context in contexts:
                for trajectory_index, actions in enumerate(reference_actions):
                    env = DenseSemiCirclePointRobot(context, "hidden", **kwargs)
                    state, _ = env.reset(seed=int(reference_seeds[trajectory_index])); states = [state]; rewards = []
                    for action in actions:
                        state, reward, _, _, _ = env.step(action); states.append(state); rewards.append(reward)
                    env.close(); states_array = np.asarray(states, dtype=np.float32)
                    rewards_array = np.asarray(rewards, dtype=np.float32)
                    raw = transition_features(states_array[:-1], actions, rewards_array, states_array[1:])
                    normalized = (raw - arrays["normalization_mean"]) / arrays["normalization_std"]
                    for t in range(len(actions) - future_horizon + 1):
                        start = max(0, t - history_length); length = t - start
                        history = np.zeros((1, history_length, 7), dtype=np.float32)
                        if length: history[0, :length] = normalized[start:t]
                        lengths = torch.as_tensor([length]); mask = torch.arange(history_length)[None, :] < lengths[:, None]
                        latent = model.encode(torch.from_numpy(history), lengths, mask, deterministic=True)
                        latents.append(latent.cpu().numpy())
                        assignment = "validation" if reference_assignments[trajectory_index] == 1 else "train"
                        rows.append({"split": split, "assignment": assignment, "episode_id": episode_id,
                                     "timestep": t, "context": float(context)})
                        if payload["method"] == "vae":
                            predicted = model.decoder(latent, torch.from_numpy(states_array[t:t + 1]),
                                torch.from_numpy(actions[t:t + future_horizon][None]))[1][0].cpu().numpy()
                            reward_errors.append({"method": payload["method"], "seed": payload["seed"],
                                "split": split, "context": float(context),
                                "squared_error": float(np.mean((predicted - rewards_array[t:t + future_horizon]) ** 2))})
                    episode_id += 1
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"frozen evaluation directory is nonempty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "latents.npz", latent=np.concatenate(latents), context=np.asarray([r["context"] for r in rows]),
             split=np.asarray([r["split"] for r in rows]), assignment=np.asarray([r["assignment"] for r in rows]),
             episode_id=np.asarray([r["episode_id"] for r in rows]), timestep=np.asarray([r["timestep"] for r in rows]))
    _csv(output / "latent_index.csv", rows)
    grouped_errors = []
    for split in EXPECTED_SPLITS:
        for context in EXPECTED_SPLITS[split]:
            selected = [r["squared_error"] for r in reward_errors if r["split"] == split and r["context"] == context]
            if selected:
                grouped_errors.append({"method": payload["method"], "seed": payload["seed"], "split": split,
                                       "context": context, "reward_mse": float(np.mean(selected))})
    _csv(output / "reward_predictions.csv", grouped_errors)
    (output / "provenance.json").write_text(json.dumps({"method": payload["method"],
        "checkpoint": str(Path(checkpoint).resolve()), "checkpoint_checksum": payload["checkpoint_checksum"],
        "dataset_checksum": metadata["dataset_checksum"]}, indent=2, sort_keys=True) + "\n")
    return output


def _write_run_artifacts(output: Path, config: dict[str, Any], metadata: dict[str, Any], method: str,
                         seed: int, model: torch.nn.Module, rows: list[dict[str, Any]],
                         negative_rows: list[dict[str, Any]]) -> None:
    with (output / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    provenance = {"source_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_root(), check=True,
                    capture_output=True, text=True).stdout.strip(), "method": method, "seed": seed,
                  "dataset_checksum": metadata["dataset_checksum"], "parameter_counts": parameter_counts(model)}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    _csv(output / "losses.csv", rows)
    _csv(output / "negative_pair_provenance.csv", negative_rows)
    best = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    record = {"metric": "total_validation_objective", "selection_scope": "held-out training-context trajectories only",
              "best_update": best["update"], "best_value": best["validation_loss"], "ood_used": False}
    (output / "checkpoint_selection.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    (output / "checkpoint_manifest.json").write_text(json.dumps({name: file_sha256(output / name)
        for name in ("best.pt", "final.pt")}, indent=2, sort_keys=True) + "\n")
    (output / "run.log").write_text(f"COMPLETE method={method} seed={seed} updates={rows[-1]['update']}\n")


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8"); return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed(seed: int, deterministic: bool) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic)
