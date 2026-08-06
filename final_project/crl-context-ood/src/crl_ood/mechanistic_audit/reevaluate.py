"""Evaluation-only reevaluation of completed goal-pilot checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from crl_ood.goal_pilot.spec import EXPECTED_SPLITS

SPECIALISTS = {
    "specialist_negative": -0.6,
    "specialist_center": 0.0,
    "specialist_positive": 0.6,
}


def evaluation_seeds(episodes: int = 100, offset: int = 5_000_000) -> list[int]:
    if episodes != 100:
        raise ValueError("The real checkpoint audit requires exactly 100 episodes")
    return [offset + index for index in range(episodes)]


def reevaluate_checkpoints(
    results_root: str | Path,
    output_dir: str | Path,
    *,
    episodes: int = 100,
    seed_offset: int = 5_000_000,
) -> dict[str, Path]:
    """Load existing models and evaluate them; never train or alter run trees."""
    from stable_baselines3 import PPO

    from crl_ood.goal_pilot.environment import make_goal_pendulum_env

    root, output = Path(results_root), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seeds = evaluation_seeds(episodes, seed_offset)
    _write_csv(output / "evaluation_seeds.csv", [
        {"episode_index": index, "episode_seed": seed, "seed_set": "mechanistic_audit_v1"}
        for index, seed in enumerate(seeds)
    ])

    specialist_rows: list[dict[str, Any]] = []
    contextual_rows: list[dict[str, Any]] = []
    runs = root / "runs"
    for kind, goal in SPECIALISTS.items():
        for training_seed in (0, 1):
            run_id = _run_id(kind, "hidden", training_seed)
            checkpoint = runs / run_id / "model.zip"
            model = PPO.load(checkpoint, device="cpu")
            specialist_rows.extend(_rollouts(
                model, make_goal_pendulum_env, run_id=run_id, kind=kind,
                method="hidden", training_seed=training_seed, split="own_goal",
                goal=goal, episode_seeds=seeds, checkpoint=checkpoint,
            ))

    for method in ("hidden", "oracle"):
        for training_seed in (0, 1):
            run_id = _run_id("contextual", method, training_seed)
            checkpoint = runs / run_id / "model.zip"
            model = PPO.load(checkpoint, device="cpu")
            for split in ("train", "id_test"):
                for goal in EXPECTED_SPLITS[split]:
                    contextual_rows.extend(_rollouts(
                        model, make_goal_pendulum_env, run_id=run_id,
                        kind="contextual", method=method,
                        training_seed=training_seed, split=split, goal=goal,
                        episode_seeds=seeds, checkpoint=checkpoint,
                    ))

    paths = {
        "seeds": output / "evaluation_seeds.csv",
        "specialist_episodes": output / "specialist_own_goal_episode_returns.csv",
        "specialist_summary": output / "specialist_own_goal_summary.csv",
        "contextual_episodes": output / "contextual_train_id_episode_returns.csv",
        "contextual_summary": output / "contextual_train_id_summary.csv",
    }
    _write_csv(paths["specialist_episodes"], specialist_rows)
    _write_csv(paths["specialist_summary"], _summary(specialist_rows))
    _write_csv(paths["contextual_episodes"], contextual_rows)
    _write_csv(paths["contextual_summary"], _summary(contextual_rows))
    return paths


def _run_id(kind: str, method: str, seed: int) -> str:
    if kind == "contextual":
        return f"contextual__all_train__{method}__seed_{seed}"
    tokens = {"specialist_negative": "goal_neg_0p6",
              "specialist_center": "goal_zero_0p0",
              "specialist_positive": "goal_pos_0p6"}
    return f"{kind}__{tokens[kind]}__{method}__seed_{seed}"


def _rollouts(
    model: Any, env_factory: Any, *, run_id: str, kind: str, method: str,
    training_seed: int, split: str, goal: float, episode_seeds: list[int],
    checkpoint: Path,
) -> list[dict[str, Any]]:
    env = env_factory({0: goal}, method, training_seed,
                      normalization=(0.0, 0.6), static_context=True)
    rows = []
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    try:
        for index, episode_seed in enumerate(episode_seeds):
            observation, _ = env.reset(seed=episode_seed)
            terminated = truncated = False
            episode_return = 0.0
            episode_length = 0
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                episode_length += 1
            rows.append({
                "run_id": run_id, "kind": kind, "method": method,
                "training_seed": training_seed, "split": split,
                "target_angle": goal, "episode_index": index,
                "episode_seed": episode_seed, "return": episode_return,
                "episode_length": episode_length,
                "termination_type": "terminated" if terminated and not truncated
                else "truncated" if truncated and not terminated
                else "terminated_and_truncated",
                "deterministic": True, "checkpoint_sha256": digest,
            })
    finally:
        env.close()
    return rows


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = ("run_id", "kind", "method", "training_seed", "split", "target_angle")
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in fields)].append(float(row["return"]))
    return [
        dict(zip(fields, key, strict=True), episodes=len(values),
             mean_return=statistics.fmean(values),
             std_return=statistics.pstdev(values),
             min_return=min(values), max_return=max(values))
        for key, values in sorted(grouped.items())
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results/goal_pilot"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/goal_pilot_mechanistic_audit/evaluation"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-offset", type=int, default=5_000_000)
    args = parser.parse_args()
    reevaluate_checkpoints(args.results_root, args.output_dir,
                           episodes=args.episodes, seed_offset=args.seed_offset)


if __name__ == "__main__":
    main()
