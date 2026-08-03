"""Dry-run, one-job, or resumable sequential goal-pilot execution."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from crl_ood.goal_pilot.matrix import (
    GoalPilotJob,
    RunState,
    build_goal_pilot_matrix,
    inspect_run,
    resolved_job_config,
)
from crl_ood.goal_pilot.spec import SPLIT_ORDER, goal_normalization, goal_splits

EPISODE_FIELDS = (
    "run_id", "method", "seed", "kind", "split", "context_id", "target_angle",
    "normalized_target_angle", "episode_index", "episode_seed", "return",
    "episode_length", "termination_type",
)
CONTEXT_FIELDS = (
    "run_id", "method", "seed", "kind", "split", "context_id", "target_angle",
    "normalized_target_angle", "episodes", "mean_return", "std_return",
)


def build_evaluation_plan(job: GoalPilotJob) -> list[dict[str, Any]]:
    """Build a deterministic plan paired across methods for a given seed."""
    splits = goal_splits(job.config)
    episodes = int(job.config["evaluation"]["episodes_per_context"])
    offset = int(job.config["evaluation"]["seed_offset"])
    plan: list[dict[str, Any]] = []
    for split_index, split in enumerate(SPLIT_ORDER):
        for context_id, target in splits[split].items():
            for episode_index in range(episodes):
                plan.append(
                    {
                        "run_id": job.job_id,
                        "method": job.mode,
                        "seed": job.seed,
                        "kind": job.kind,
                        "split": split,
                        "context_id": context_id,
                        "target_angle": target,
                        "episode_index": episode_index,
                        "episode_seed": offset + job.seed + split_index * 100_000 + context_id * 1_000 + episode_index,
                    }
                )
    return plan


def train_one(job: GoalPilotJob, *, overwrite: bool = False) -> Path:
    """Train and evaluate one atomic run; imported RL dependencies stay local."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.monitor import Monitor

    from crl_ood.goal_pilot.environment import make_goal_pendulum_env
    from crl_ood.utils.metadata import collect_metadata
    from crl_ood.utils.seeding import seed_everything

    run_dir = job.output_dir
    if run_dir.exists() and any(run_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Run directory already contains artifacts: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    _log(log_path, f"START {job.job_id}")

    deterministic_torch = bool(job.config["reproducibility"]["deterministic_torch"])
    seed_everything(job.seed, deterministic_torch=deterministic_torch)
    all_splits = goal_splits(job.config)
    normalization = goal_normalization(all_splits["train"])
    plan = build_evaluation_plan(job)
    _write_provenance(job, run_dir, collect_metadata())
    _write_context_artifacts(job, run_dir, all_splits, normalization, plan)

    train_goals = {index: goal for index, goal in enumerate(job.training_goals)}
    base_env = make_goal_pendulum_env(
        train_goals, job.mode, job.seed, normalization=normalization
    )
    env = Monitor(base_env, filename=str(run_dir / "sb3_monitor.csv"), override_existing=True)
    training = job.config["training"]
    model = PPO(
        "MlpPolicy", env,
        learning_rate=float(training["learning_rate"]),
        n_steps=int(training["n_steps"]),
        batch_size=int(training["batch_size"]),
        n_epochs=int(training["n_epochs"]),
        gamma=float(training["gamma"]),
        gae_lambda=float(training["gae_lambda"]),
        seed=job.seed,
        device=str(training["device"]),
        verbose=int(training.get("verbose", 0)),
    )
    model.set_logger(configure(str(run_dir / "sb3_logs"), ["csv"]))
    model.learn(total_timesteps=int(training["total_timesteps"]), progress_bar=False)
    model.save(run_dir / "model")
    _write_training_metrics(job, run_dir, env)
    env.close()

    seed_everything(job.seed, deterministic_torch=deterministic_torch)
    _evaluate(model, job, run_dir, all_splits, normalization, plan)
    _log(log_path, f"COMPLETE {job.job_id}")
    return run_dir


def _evaluate(model: Any, job: GoalPilotJob, run_dir: Path,
              splits: dict[str, dict[int, float]], normalization: tuple[float, float],
              plan: list[dict[str, Any]]) -> None:
    from crl_ood.goal_pilot.environment import make_goal_pendulum_env

    deterministic = bool(job.config["evaluation"].get("deterministic", True))
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in plan:
        grouped[(str(row["split"]), int(row["context_id"]))].append(row)
    rows: list[dict[str, Any]] = []
    for (split, context_id), episodes in grouped.items():
        target = splits[split][context_id]
        env = make_goal_pendulum_env(
            {0: target}, job.mode, job.seed,
            normalization=normalization, static_context=True,
        )
        for planned in episodes:
            observation, _ = env.reset(seed=int(planned["episode_seed"]))
            terminated = truncated = False
            episode_return = 0.0
            episode_length = 0
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=deterministic)
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                episode_length += 1
            rows.append(
                {
                    **planned,
                    "normalized_target_angle": (target - normalization[0]) / normalization[1],
                    "return": episode_return,
                    "episode_length": episode_length,
                    "termination_type": _termination_type(terminated, truncated),
                }
            )
        env.close()
    _write_csv(run_dir / "episode_returns.csv", EPISODE_FIELDS, rows)
    _write_context_returns(run_dir / "context_returns.csv", rows)


def _write_provenance(job: GoalPilotJob, run_dir: Path, metadata: dict[str, Any]) -> None:
    with (run_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved_job_config(job), handle, sort_keys=False)
    (run_dir / "seed.txt").write_text(f"{job.seed}\n", encoding="ascii")
    with (run_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_context_artifacts(
    job: GoalPilotJob, run_dir: Path, splits: dict[str, dict[int, float]],
    normalization: tuple[float, float], plan: list[dict[str, Any]],
) -> None:
    center, scale = normalization
    manifest = {
        "context_feature": "target_angle",
        "normalization": {"source": "training_goal_extrema", "center": center, "scale": scale},
        "training_goals": list(job.training_goals),
        "splits": {
            split: [
                {
                    "context_id": context_id,
                    "target_angle": target,
                    "normalized_target_angle": (target - center) / scale,
                }
                for context_id, target in contexts.items()
            ]
            for split, contexts in splits.items()
        },
    }
    with (run_dir / "contexts.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False)
    context_rows = [
        {
            "run_id": job.job_id, "method": job.mode, "kind": job.kind,
            "split": split, "context_id": context_id, "target_angle": target,
            "normalized_target_angle": (target - center) / scale,
        }
        for split, contexts in splits.items() for context_id, target in contexts.items()
    ]
    _write_csv(run_dir / "contexts.csv", tuple(context_rows[0]), context_rows)
    fields = ("run_id", "method", "seed", "kind", "split", "context_id", "target_angle", "episode_index", "episode_seed")
    _write_csv(run_dir / "evaluation_plan.csv", fields, plan)


def _write_training_metrics(job: GoalPilotJob, run_dir: Path, monitor: Any) -> None:
    fields = ("run_id", "method", "seed", "kind", "environment_steps", "episode_index", "episode_return", "episode_length")
    rows = []
    steps = 0
    for index, (episode_return, episode_length) in enumerate(
        zip(monitor.get_episode_rewards(), monitor.get_episode_lengths(), strict=True)
    ):
        steps += int(episode_length)
        rows.append(
            {"run_id": job.job_id, "method": job.mode, "seed": job.seed,
             "kind": job.kind, "environment_steps": steps, "episode_index": index,
             "episode_return": episode_return, "episode_length": episode_length}
        )
    if not rows:
        raise RuntimeError("Smoke/real budget must complete at least one monitored episode")
    _write_csv(run_dir / "training_metrics.csv", fields, rows)


def _write_context_returns(path: Path, rows: list[dict[str, Any]]) -> None:
    import statistics
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    key_fields = CONTEXT_FIELDS[:8]
    for row in rows:
        grouped[tuple(row[field] for field in key_fields)].append(float(row["return"]))
    summaries = []
    for key, returns in grouped.items():
        summaries.append(dict(zip(CONTEXT_FIELDS, (*key, len(returns), statistics.fmean(returns), statistics.pstdev(returns)), strict=True)))
    _write_csv(path, CONTEXT_FIELDS, summaries)


def _termination_type(terminated: bool, truncated: bool) -> str:
    if terminated and truncated:
        return "terminated_and_truncated"
    if terminated:
        return "terminated"
    if truncated:
        return "truncated"
    raise RuntimeError("Episode did not terminate or truncate")


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _log(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-config", type=Path, default=Path("configs/goal_pilot/matrix.yaml"))
    parser.add_argument("--job-id", help="Run exactly one unique atomic job")
    parser.add_argument("--dry-run", action="store_true", help="Print selected jobs without importing RL dependencies")
    parser.add_argument("--resume", action="store_true", help="Skip only validated complete jobs")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace selected atomic outputs")
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    jobs = build_goal_pilot_matrix(args.matrix_config)
    if args.job_id:
        jobs = [job for job in jobs if job.job_id == args.job_id]
        if not jobs:
            parser.error(f"unknown --job-id {args.job_id!r}")
    statuses = [(job, inspect_run(job)) for job in jobs]
    _print_plan(statuses)
    if args.dry_run:
        return
    runnable: list[GoalPilotJob] = []
    for job, status in statuses:
        if args.resume:
            if status.state is RunState.COMPLETE:
                print(f"SKIP validated complete: {job.job_id}", flush=True)
                continue
            if status.state is RunState.PARTIAL:
                raise SystemExit(f"Refusing ambiguous partial directory for {job.job_id}: {status.detail}")
        elif status.state is not RunState.PENDING and not args.overwrite:
            raise SystemExit(
                f"Refusing existing output for {job.job_id} ({status.state.value}): {status.detail}. "
                "Use --resume or explicitly --overwrite."
            )
        runnable.append(job)
    for index, job in enumerate(runnable, start=1):
        print(f"START {index}/{len(runnable)} {job.job_id}", flush=True)
        train_one(job, overwrite=args.overwrite)
        status = inspect_run(job)
        if status.state is not RunState.COMPLETE:
            raise RuntimeError(f"Job returned without a validated complete run: {job.job_id}: {status.detail}")
        print(f"COMPLETE {job.job_id}: {job.output_dir}", flush=True)


def _print_plan(statuses: list[tuple[GoalPilotJob, Any]]) -> None:
    print("job_id\ttimesteps\tmode\tseed\troles\tstate\toutput_dir")
    for job, status in statuses:
        print(f"{job.job_id}\t{job.total_timesteps}\t{job.mode}\t{job.seed}\t{','.join(job.roles)}\t{status.state.value}\t{job.output_dir}")
    print(f"jobs={len(statuses)} unique_atomic_runs={len(statuses)} concurrency=1")


if __name__ == "__main__":
    main()
