"""Dry-run, one-job, or resumable sequential PointRobot gate execution."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from crl_ood.pointrobot_gate.matrix import PointRobotJob, RunState, build_matrix, inspect_run, resolved_config
from crl_ood.pointrobot_gate.spec import SPLIT_ORDER, context_record, context_splits


def build_evaluation_plan(job: PointRobotJob) -> list[dict[str, Any]]:
    splits = context_splits(job.config)
    episodes = int(job.config["evaluation"]["episodes_per_context"])
    offset = int(job.config["evaluation"]["seed_offset"])
    rows = []
    for split_index, split in enumerate(SPLIT_ORDER):
        for context_id, angle in splits[split].items():
            for episode_index in range(episodes):
                rows.append({
                    "run_id": job.job_id, "method": job.mode, "kind": job.kind, "seed": job.seed,
                    **context_record(split, context_id, angle), "episode_index": episode_index,
                    "episode_seed": offset + job.seed + split_index * 100_000 + context_id * 1_000 + episode_index,
                })
    return rows


def train_one(job: PointRobotJob, *, overwrite: bool = False) -> Path:
    """Train and evaluate one run. RL/environment imports remain local."""
    if job.output_dir.exists() and any(job.output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Run directory already contains artifacts: {job.output_dir}")
        shutil.rmtree(job.output_dir)
    from stable_baselines3 import PPO
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.monitor import Monitor
    from crl_ood.pointrobot_gate.environment import DenseSemiCirclePointRobot
    from crl_ood.utils.metadata import collect_metadata
    from crl_ood.utils.seeding import seed_everything

    run_dir = job.output_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    _log(run_dir / "run.log", f"START {job.job_id}")
    deterministic_torch = bool(job.config["reproducibility"]["deterministic_torch"])
    seed_everything(job.seed, deterministic_torch=deterministic_torch)
    plan = build_evaluation_plan(job)
    env_kwargs = _env_kwargs(job.config)
    probe_env = DenseSemiCirclePointRobot(job.training_goals, job.mode, **env_kwargs)
    _write_provenance(job, run_dir, collect_metadata(), probe_env.environment_spec, plan)
    env = Monitor(probe_env, filename=str(run_dir / "sb3_monitor.csv"), override_existing=True)
    training = job.config["training"]
    model = PPO(
        "MlpPolicy", env, learning_rate=float(training["learning_rate"]),
        n_steps=int(training["n_steps"]), batch_size=int(training["batch_size"]),
        n_epochs=int(training["n_epochs"]), gamma=float(training["gamma"]),
        gae_lambda=float(training["gae_lambda"]), seed=job.seed,
        device=str(training["device"]), verbose=int(training.get("verbose", 0)),
    )
    model.set_logger(configure(str(run_dir / "sb3_logs"), ["csv"]))
    model.learn(total_timesteps=int(training["total_timesteps"]), progress_bar=False)
    model.save(run_dir / "model")
    _training_metrics(job, run_dir, env)
    env.close()
    seed_everything(job.seed, deterministic_torch=deterministic_torch)
    _evaluate(model, job, run_dir, plan, env_kwargs)
    _log(run_dir / "run.log", f"COMPLETE {job.job_id}")
    return run_dir


def _env_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    env = config["environment"]
    return {key: env[key] for key in (
        "goal_radius", "start_position", "reset_noise", "step_scale", "position_limit",
        "horizon", "action_penalty", "success_threshold",
    )}


def _evaluate(model: Any, job: PointRobotJob, run_dir: Path, plan: list[dict[str, Any]], kwargs: dict[str, Any]) -> None:
    from crl_ood.pointrobot_gate.environment import DenseSemiCirclePointRobot
    rows = []
    for planned in plan:
        env = DenseSemiCirclePointRobot(float(planned["goal_angle"]), job.mode, **kwargs)
        observation, _ = env.reset(seed=int(planned["episode_seed"]))
        total = 0.0
        terminated = truncated = False
        terminal_info: dict[str, Any] = {}
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=bool(job.config["evaluation"]["deterministic"]))
            observation, reward, terminated, truncated, terminal_info = env.step(action)
            total += float(reward)
        rows.append({**planned, "return": total, "episode_length": env.horizon,
                     "termination_type": "truncated", **terminal_info})
        env.close()
    _csv(run_dir / "episode_returns.csv", rows)
    summaries = _summaries(rows)
    _csv(run_dir / "context_returns.csv", summaries)
    _csv(run_dir / "success_metrics.csv", [{k: row[k] for k in (
        "run_id", "method", "kind", "seed", "split", "context_id", "goal_angle", "goal_cos",
        "goal_sin", "goal_x", "goal_y", "episodes", "success_rate") } for row in summaries])
    _csv(run_dir / "distance_metrics.csv", [{k: row[k] for k in (
        "run_id", "method", "kind", "seed", "split", "context_id", "goal_angle", "goal_cos",
        "goal_sin", "goal_x", "goal_y", "episodes", "mean_final_distance", "mean_minimum_distance") } for row in summaries])


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    key_fields = ("run_id", "method", "kind", "seed", "split", "context_id", "goal_angle", "goal_cos", "goal_sin", "goal_x", "goal_y")
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in key_fields)].append(row)
    output = []
    for key, values in grouped.items():
        output.append({**dict(zip(key_fields, key, strict=True)), "episodes": len(values),
                       "mean_return": statistics.fmean(float(x["return"]) for x in values),
                       "std_return": statistics.pstdev(float(x["return"]) for x in values),
                       "success_rate": statistics.fmean(float(bool(x["success"])) for x in values),
                       "mean_final_distance": statistics.fmean(float(x["final_distance"]) for x in values),
                       "mean_minimum_distance": statistics.fmean(float(x["minimum_distance"]) for x in values),
                       "mean_first_success_timestep": _optional_mean(x["first_success_timestep"] for x in values)})
    return output


def _optional_mean(values: Any) -> float | str:
    present = [float(x) for x in values if x is not None and x != ""]
    return statistics.fmean(present) if present else ""


def _write_provenance(job: PointRobotJob, run_dir: Path, metadata: dict[str, Any], env_spec: dict[str, Any], plan: list[dict[str, Any]]) -> None:
    with (run_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved_config(job), handle, sort_keys=False)
    (run_dir / "seed.txt").write_text(f"{job.seed}\n", encoding="ascii")
    splits = context_splits(job.config)
    contexts = {split: [context_record(split, cid, angle) for cid, angle in values.items()] for split, values in splits.items()}
    metadata = dict(metadata)
    metadata["pointrobot_context_metadata"] = {
        "training_goal_angles": list(job.training_goals),
        "splits": contexts,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    provenance = {"implementation": "independently written", "inspiration": "ContraBAR Semi-Circle PointRobot",
                  "differences": ["dense reward", "fixed-horizon single episode", "state observations", "explicit continuous ID/OOD splits", "later frozen-encoder benchmark"]}
    (run_dir / "source_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (run_dir / "environment_spec.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(env_spec, handle, sort_keys=False)
    with (run_dir / "contexts.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"training_goals": list(job.training_goals), "splits": contexts}, handle, sort_keys=False)
    context_rows = [{"run_id": job.job_id, "method": job.mode, "kind": job.kind, **row} for values in contexts.values() for row in values]
    _csv(run_dir / "contexts.csv", context_rows)
    _csv(run_dir / "evaluation_plan.csv", plan)


def _training_metrics(job: PointRobotJob, run_dir: Path, monitor: Any) -> None:
    rows, steps = [], 0
    for index, (episode_return, length) in enumerate(zip(monitor.get_episode_rewards(), monitor.get_episode_lengths(), strict=True)):
        steps += int(length)
        rows.append({"run_id": job.job_id, "method": job.mode, "kind": job.kind, "seed": job.seed,
                     "environment_steps": steps, "episode_index": index,
                     "episode_return": episode_return, "episode_length": length})
    if not rows:
        raise RuntimeError("Budget must complete at least one 50-step episode")
    _csv(run_dir / "training_metrics.csv", rows)


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty artifact {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _log(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n"); handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-config", type=Path, default=Path("configs/pointrobot_gate/matrix.yaml"))
    parser.add_argument("--job-id")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    jobs = build_matrix(args.matrix_config)
    if args.job_id:
        jobs = [job for job in jobs if job.job_id == args.job_id]
        if not jobs:
            parser.error(f"unknown --job-id {args.job_id!r}")
    statuses = [(job, inspect_run(job)) for job in jobs]
    print("job_id\ttimesteps\tmode\tseed\troles\tstate\toutput_dir", flush=True)
    for job, status in statuses:
        print(f"{job.job_id}\t{job.total_timesteps}\t{job.mode}\t{job.seed}\t{','.join(job.roles)}\t{status.state.value}\t{job.output_dir}", flush=True)
    print(f"jobs={len(jobs)} unique_atomic_runs={len(jobs)} concurrency=1", flush=True)
    if args.dry_run:
        return
    runnable = []
    for job, status in statuses:
        if args.resume and status.state is RunState.COMPLETE:
            print(f"SKIP validated complete: {job.job_id}", flush=True); continue
        if args.resume and status.state is RunState.PARTIAL:
            raise SystemExit(f"Refusing ambiguous partial directory for {job.job_id}: {status.detail}")
        if not args.resume and status.state is not RunState.PENDING and not args.overwrite:
            raise SystemExit(f"Refusing existing output for {job.job_id}: {status.detail}")
        runnable.append(job)
    for index, job in enumerate(runnable, 1):
        print(f"START {index}/{len(runnable)} {job.job_id}", flush=True)
        train_one(job, overwrite=args.overwrite)
        status = inspect_run(job)
        if status.state is not RunState.COMPLETE:
            raise RuntimeError(f"Job incomplete after execution: {job.job_id}: {status.detail}")
        print(f"COMPLETE {job.job_id}", flush=True)


if __name__ == "__main__":
    main()
