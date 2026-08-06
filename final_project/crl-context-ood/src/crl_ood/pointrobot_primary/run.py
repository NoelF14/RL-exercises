"""Execution entry points for the frozen primary matrix (heavy imports are lazy)."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from crl_ood.pointrobot_primary.spec import (LEARNED_METHODS, SPLITS, PrimaryJob,
    build_downstream_jobs, build_encoder_jobs, load_spec, sha256_file, validate_encoder_runs,
    validate_timestep_budget)
from crl_ood.utils.paths import project_root

DEFAULT_SPEC = "configs/pointrobot_primary/spec.yaml"


def _encoder_config(spec: dict[str, Any], root: Path) -> dict[str, Any]:
    source_path = root / spec["dataset"]["source_config"]
    source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    dataset, architecture, optimizer = source["dataset"], spec["encoder"]["architecture"], spec["encoder"]["optimizer"]
    checks = (
        (dataset["budgets"]["full"]["trajectories_per_context"], spec["dataset"]["trajectories_per_context"], "full budget"),
        (source["encoder"]["history_length"], architecture["history_length"], "H"),
        (source["encoder"]["future_horizon"], architecture["future_horizon"], "K"),
        (source["encoder"]["hidden_size"], architecture["hidden_size"], "hidden size"),
        (source["encoder"]["latent_dim"], architecture["latent_size"], "latent size"),
        (source["encoder"]["max_updates"], optimizer["max_updates"], "updates"),
        (dataset["behavior_policy"], spec["dataset"]["behavior_policy"], "behavior policy"),
    )
    for actual, expected, label in checks:
        if actual != expected:
            raise ValueError(f"source encoder config no longer matches frozen primary {label}")
    exact_dataset = {
        "train_contexts": spec["dataset"]["train_contexts"],
        "orthogonal_prefix": spec["dataset"]["orthogonal_prefix"],
        "validation_fraction": spec["dataset"]["validation_fraction"],
        "split_seed": spec["dataset"]["split_seed"],
    }
    if any(dataset.get(key) != value for key, value in exact_dataset.items()):
        raise ValueError("source dataset collection/splitting fields differ from the frozen primary specification")
    exact_encoder = {"learning_rate": 0.001, "batch_size": 64, "validation_interval": 200,
                     "gradient_clip_norm": 10.0, "optimizer": "Adam"}
    if any(source["encoder"].get(key) != value for key, value in exact_encoder.items()):
        raise ValueError("source encoder optimizer fields differ from the frozen primary specification")
    if source.get("vae") != spec["encoder"]["objectives"]["vae"]:
        raise ValueError("source VAE objective differs from the frozen primary specification")
    contrastive = spec["encoder"]["objectives"]["contrastive"]
    if source.get("contrastive", {}).get("temperature") != contrastive["temperature"] or \
       source.get("contrastive", {}).get("negative_mode") != contrastive["negative_mode"]:
        raise ValueError("source contrastive objective differs from the frozen primary specification")
    source["experiment"] = {**source["experiment"], "name": "pointrobot_primary",
                            "results_dir": spec["experiment"]["results_dir"]}
    source["pilot"] = {"methods": list(spec["encoder"]["methods"]),
                       "seeds": list(spec["encoder"]["seeds"]), "dataset_budget": "full"}
    return source


def dataset_command(spec: dict[str, Any], root: Path, dry_run: bool) -> None:
    config = _encoder_config(spec, root)
    destination = (root / spec["dataset"]["checksum_artifact"]).parent
    if dry_run:
        per_context = int(spec["dataset"]["trajectories_per_context"])
        horizon = int(config["dataset"]["horizon"])
        print(json.dumps({"status": "planned_not_collected", "budget": "full",
            "destination": str(destination), "training_contexts": spec["dataset"]["train_contexts"],
            "episodes": per_context * 5, "transitions": per_context * 5 * horizon,
            "checksum_artifact": spec["dataset"]["checksum_artifact"]}, indent=2))
        return
    from crl_ood.pointrobot_encoders.dataset import collect_arrays, save_dataset
    arrays, metadata = collect_arrays(config, "full")
    save_dataset(destination, arrays, metadata)
    print(f"collected immutable full dataset: {destination} checksum={metadata['dataset_checksum']}")


def encoder_matrix_command(spec: dict[str, Any], root: Path, dry_run: bool, resume: bool) -> None:
    jobs = build_encoder_jobs(spec, root)
    if dry_run:
        for job in jobs:
            print(f"{job.method} encoder_seed={job.encoder_seed} updates=20000 dataset_sha256={job.dataset_checksum} output={job.output_dir}")
        print(f"encoder jobs: {len(jobs)}")
        return
    from crl_ood.pointrobot_encoders.training import train_encoder
    config = _encoder_config(spec, root)
    for job in jobs:
        if job.output_dir.joinpath("run.log").is_file() and job.output_dir.joinpath("run.log").read_text().startswith("COMPLETE"):
            print(f"skip complete: {job.output_dir}")
            continue
        train_encoder(config, job.dataset_dir, job.method, job.encoder_seed, job.output_dir,
                      resume=resume and job.output_dir.exists())
        print(f"complete: {job.output_dir}")


def validate_encoders_command(spec: dict[str, Any], root: Path) -> None:
    rows = validate_encoder_runs(spec, root)
    _validate_checkpoint_payloads(rows)
    print(json.dumps({"status": "PASS", "encoder_runs": len(rows),
                      "dataset_checksum": rows[0]["dataset_checksum"]}, indent=2))


def _validate_checkpoint_payloads(rows: list[dict[str, Any]]) -> None:
    """Verify the dataset/method/seed embedded inside every actual checkpoint."""
    from crl_ood.pointrobot_encoders.training import load_frozen_checkpoint
    for row in rows:
        _, payload = load_frozen_checkpoint(row["checkpoint"], row["method"], row["dataset_checksum"])
        if int(payload.get("seed", -1)) != int(row["encoder_seed"]):
            raise ValueError("encoder seed embedded in checkpoint does not match its paired run")
        if int(payload.get("parameter_counts", {}).get("downstream_retained", -1)) != 14536:
            raise ValueError("checkpoint does not retain the frozen 14536-parameter encoder")
        config = payload.get("config", {})
        encoder = config.get("encoder", {})
        expected = {"history_length": 5, "future_horizon": 5, "transition_dim": 7,
                    "hidden_size": 64, "latent_dim": 8, "num_gru_layers": 1, "max_updates": 20000}
        if any(int(encoder.get(key, -1)) != value for key, value in expected.items()):
            raise ValueError("checkpoint encoder specification differs from the frozen primary specification")
        if row["method"] == "vae":
            objective = config.get("vae", {})
            if any(float(objective.get(key, math.nan)) != value for key, value in
                   {"state_loss_weight": 1.0, "reward_loss_weight": 1.0, "kl_weight": 0.001}.items()):
                raise ValueError("checkpoint VAE objective differs from the frozen primary specification")
        else:
            objective = config.get("contrastive", {})
            if objective.get("negative_mode") != "reward_relabel" or float(objective.get("temperature", math.nan)) != 0.1:
                raise ValueError("checkpoint contrastive objective differs from the frozen primary specification")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty required artifact {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _run_job(job: PrimaryJob, spec: dict[str, Any], root: Path) -> Path:
    if job.output_dir.exists() and any(job.output_dir.iterdir()):
        raise FileExistsError(f"downstream run is nonempty/partial: {job.output_dir}")
    job.output_dir.mkdir(parents=True, exist_ok=False)
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from crl_ood.pointrobot_encoders.wrapper import make_policy_env

    gate = yaml.safe_load((root / "configs/pointrobot_gate/gate.yaml").read_text(encoding="utf-8"))
    env_cfg = gate["environment"]
    env_kwargs = {key: env_cfg[key] for key in ("goal_radius", "start_position", "reset_noise", "step_scale",
        "position_limit", "horizon", "action_penalty", "success_threshold")}
    env = make_policy_env(job.method, SPLITS["train"], env_kwargs, checkpoint=job.checkpoint,
                          dataset_checksum=job.dataset_checksum if job.checkpoint else None)
    ppo = spec["policy"]["ppo"]

    class ProgressCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(); self.rows: list[dict[str, Any]] = []
        def _on_step(self) -> bool:
            return True
        def _on_rollout_end(self) -> None:
            returns = [float(item["r"]) for item in self.model.ep_info_buffer]
            self.rows.append({"timesteps": int(self.num_timesteps),
                              "mean_recent_episode_return": float(np.mean(returns)) if returns else math.nan,
                              "recent_episode_count": len(returns)})

    callback = ProgressCallback()
    model = PPO("MlpPolicy", env, seed=job.policy_seed, learning_rate=float(ppo["learning_rate"]),
        n_steps=job.rollout_quantum, batch_size=int(ppo["batch_size"]), n_epochs=int(ppo["n_epochs"]),
        gamma=float(ppo["gamma"]), gae_lambda=float(ppo["gae_lambda"]), device=str(ppo["device"]), verbose=0)
    model.learn(total_timesteps=job.requested_timesteps, progress_bar=False, callback=callback)
    actual = int(model.num_timesteps)
    validate_timestep_budget(job.requested_timesteps, actual, job.rollout_quantum)
    model.save(job.output_dir / "model")
    episode_rows, context_rows = _evaluate(model, job, spec, env_kwargs, make_policy_env)
    _write_csv(job.output_dir / "evaluation_episodes.csv", episode_rows)
    _write_csv(job.output_dir / "context_metrics.csv", context_rows)
    _write_csv(job.output_dir / "training_progress.csv", callback.rows)
    provenance = {"method": job.method, "encoder_seed": job.encoder_seed, "policy_seed": job.policy_seed,
        "requested_timesteps": job.requested_timesteps, "actual_complete_rollout_timesteps": actual,
        "rollout_quantum": job.rollout_quantum, "dataset_checksum": job.dataset_checksum,
        "encoder_checkpoint_path": str(job.checkpoint) if job.checkpoint else None,
        "encoder_checkpoint_sha256": job.checkpoint_sha256, "source_commit": spec["experiment"]["source_commit"],
        "configuration_checksum": spec["_configuration_checksum"],
        "selection_split": "training contexts only",
        "ood_role": "descriptive/scientific evaluation only",
        "evaluation_episodes_per_context": int(spec["evaluation"]["episodes_per_context"]),
        "evaluation_deterministic": bool(spec["evaluation"]["deterministic"])}
    (job.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (job.output_dir / "run.log").write_text("COMPLETE\n", encoding="utf-8")
    env.close()
    return job.output_dir


def _evaluate(model: Any, job: PrimaryJob, spec: dict[str, Any], env_kwargs: dict[str, Any],
              make_policy_env: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episodes, contexts = [], []
    count = int(spec["evaluation"]["episodes_per_context"])
    deterministic = bool(spec["evaluation"]["deterministic"])
    offset = int(spec["evaluation"]["seed_offset"])
    global_context_index = 0
    for split, goals in SPLITS.items():
        for goal in goals:
            current = []
            for episode in range(count):
                env = make_policy_env(job.method, goal, env_kwargs, checkpoint=job.checkpoint,
                    dataset_checksum=job.dataset_checksum if job.checkpoint else None)
                evaluation_seed = offset + job.policy_seed * 100000 + global_context_index * count + episode
                observation, _ = env.reset(seed=evaluation_seed)
                total, done, final_info = 0.0, False, {}
                while not done:
                    action, _ = model.predict(observation, deterministic=deterministic)
                    observation, reward, terminated, truncated, info = env.step(action)
                    total += float(reward); done = bool(terminated or truncated)
                    if done: final_info = info
                row = {"method": job.method, "policy_seed": job.policy_seed, "encoder_seed": job.encoder_seed,
                    "split": split, "goal_angle": goal, "evaluation_episode": episode,
                    "evaluation_seed": evaluation_seed, "return": total,
                    "success": int(bool(final_info["success"])), "final_distance": float(final_info["final_distance"]),
                    "minimum_distance": float(final_info["minimum_distance"]),
                    "first_success_timestep": final_info["first_success_timestep"]}
                episodes.append(row); current.append(row); env.close()
            successful_times = [float(row["first_success_timestep"]) for row in current
                                if row["first_success_timestep"] not in (None, "")]
            returns = np.asarray([row["return"] for row in current], dtype=float)
            contexts.append({"method": job.method, "policy_seed": job.policy_seed, "encoder_seed": job.encoder_seed,
                "split": split, "goal_angle": goal, "mean_return": float(returns.mean()),
                "return_std": float(returns.std(ddof=1)), "success_rate": float(np.mean([r["success"] for r in current])),
                "mean_final_distance": float(np.mean([r["final_distance"] for r in current])),
                "mean_minimum_distance": float(np.mean([r["minimum_distance"] for r in current])),
                "mean_first_success_timestep": float(np.mean(successful_times)) if successful_times else math.nan,
                "evaluation_episode_count": count})
            global_context_index += 1
    return episodes, contexts


def downstream_command(spec: dict[str, Any], root: Path, dry_run: bool, resume: bool) -> None:
    _validate_checkpoint_payloads(validate_encoder_runs(spec, root))
    jobs = build_downstream_jobs(spec, root)
    if dry_run:
        for job in jobs:
            print(f"{job.method} policy_seed={job.policy_seed} encoder_seed={job.encoder_seed} "
                  f"steps={job.requested_timesteps} checkpoint={job.checkpoint} output={job.output_dir}")
        print(f"downstream jobs: {len(jobs)}")
        return
    for job in jobs:
        complete = job.output_dir.joinpath("run.log")
        if resume and complete.is_file() and complete.read_text().startswith("COMPLETE"):
            print(f"skip complete: {job.output_dir}"); continue
        _run_job(job, spec, root); print(f"complete: {job.output_dir}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default=DEFAULT_SPEC)
    commands = parser.add_subparsers(dest="command", required=True)
    dataset = commands.add_parser("dataset"); dataset.add_argument("--dry-run", action="store_true")
    encoders = commands.add_parser("encoder-matrix"); encoders.add_argument("--dry-run", action="store_true"); encoders.add_argument("--resume", action="store_true")
    commands.add_parser("validate-encoders")
    downstream = commands.add_parser("downstream"); downstream.add_argument("--dry-run", action="store_true"); downstream.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    root = project_root(); spec = load_spec(root / args.spec)
    if args.command == "dataset": dataset_command(spec, root, args.dry_run)
    elif args.command == "encoder-matrix": encoder_matrix_command(spec, root, args.dry_run, args.resume)
    elif args.command == "validate-encoders": validate_encoders_command(spec, root)
    else: downstream_command(spec, root, args.dry_run, args.resume)


if __name__ == "__main__":
    main()
