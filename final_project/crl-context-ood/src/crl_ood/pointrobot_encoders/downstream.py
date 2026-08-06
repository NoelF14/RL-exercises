"""Plan and optionally execute atomic frozen-encoder PointRobot PPO jobs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from crl_ood.pointrobot_encoders.training import file_sha256
from crl_ood.pointrobot_encoders.wrapper import make_policy_env
from crl_ood.pointrobot_gate.spec import EXPECTED_SPLITS
from crl_ood.utils.paths import project_root


@dataclass(frozen=True)
class DownstreamJob:
    method: str
    seed: int
    total_timesteps: int
    output_dir: Path
    checkpoint: Path | None
    dataset_checksum: str | None


def build_jobs(config: dict[str, Any], matrix: str, checkpoints: dict[str, str] | None,
               dataset_checksum: str | None, *, timesteps_override: int | None = None) -> list[DownstreamJob]:
    section = config[matrix]
    if matrix == "full_primary" and section["status"].startswith("unavailable"):
        raise RuntimeError(section["status"])
    root = project_root() / config["experiment"]["results_dir"] / matrix
    jobs = []
    for method in config["methods"]:
        checkpoint = Path(checkpoints[method]).resolve() if checkpoints and method in checkpoints else None
        if method in {"vae", "contrastive"} and (checkpoint is None or not dataset_checksum):
            raise ValueError(f"{method} requires an explicit checkpoint and dataset checksum")
        for seed in section["seeds"]:
            steps = int(timesteps_override if timesteps_override is not None else section["total_timesteps"])
            jobs.append(DownstreamJob(method, int(seed), steps, root / method / f"seed_{seed}",
                                      checkpoint, dataset_checksum if checkpoint else None))
    return jobs


def run_job(job: DownstreamJob, gate_config: dict[str, Any]) -> Path:
    if job.output_dir.exists() and any(job.output_dir.iterdir()):
        raise FileExistsError(f"downstream run is nonempty/partial: {job.output_dir}")
    job.output_dir.mkdir(parents=True, exist_ok=True)
    from stable_baselines3 import PPO
    env_cfg = gate_config["environment"]
    kwargs = {key: env_cfg[key] for key in ("goal_radius", "start_position", "reset_noise", "step_scale",
              "position_limit", "horizon", "action_penalty", "success_threshold")}
    env = make_policy_env(job.method, EXPECTED_SPLITS["train"], kwargs, checkpoint=job.checkpoint,
                          dataset_checksum=job.dataset_checksum)
    training = gate_config["training"]
    n_steps = min(int(training["n_steps"]), job.total_timesteps)
    batch_size = min(int(training["batch_size"]), n_steps)
    model = PPO("MlpPolicy", env, seed=job.seed, learning_rate=float(training["learning_rate"]),
                n_steps=n_steps, batch_size=batch_size, n_epochs=int(training["n_epochs"]),
                gamma=float(training["gamma"]), gae_lambda=float(training["gae_lambda"]),
                device="cpu", verbose=0)
    provenance = {"method": job.method, "seed": job.seed, "total_timesteps": job.total_timesteps,
                  "encoder_checkpoint": str(job.checkpoint) if job.checkpoint else None,
                  "encoder_checkpoint_checksum": file_sha256(job.checkpoint) if job.checkpoint else None,
                  "dataset_checksum": job.dataset_checksum, "selection_splits": ["train"],
                  "ood_selection_role": "descriptive_only"}
    (job.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    model.learn(total_timesteps=job.total_timesteps, progress_bar=False)
    model.save(job.output_dir / "model")
    rows = _evaluate(model, job, kwargs)
    with (job.output_dir / "episode_returns.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    env.close(); (job.output_dir / "run.log").write_text("COMPLETE\n")
    return job.output_dir


def _evaluate(model: Any, job: DownstreamJob, kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for split, goals in EXPECTED_SPLITS.items():
        for context_index, goal in enumerate(goals):
            env = make_policy_env(job.method, goal, kwargs, checkpoint=job.checkpoint,
                                  dataset_checksum=job.dataset_checksum)
            observation, _ = env.reset(seed=900000 + job.seed * 10000 + context_index)
            total = 0.0; done = False
            while not done:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(action)
                total += float(reward); done = terminated or truncated
            rows.append({"method": job.method, "seed": job.seed, "split": split,
                         "goal_angle": goal, "return": total})
            env.close()
    return rows


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)
