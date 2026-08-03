"""Deterministic, paired evaluation for trained Phase 0 agents."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO

from crl_ood.environments.context_splits import (
    build_context_splits,
    carl_feature_key,
)
from crl_ood.environments.factory import make_pendulum_env
from crl_ood.utils.metadata import load_config
from crl_ood.utils.seeding import seed_everything

EPISODE_FIELDS = (
    "experiment",
    "feature",
    "mode",
    "seed",
    "split",
    "context_id",
    "context_value",
    "episode",
    "episode_seed",
    "return",
    "length",
)
CONTEXT_FIELDS = (
    "experiment",
    "feature",
    "mode",
    "seed",
    "split",
    "context_id",
    "context_value",
    "episodes",
    "mean_return",
    "std_return",
)


def evaluate_model(
    model: PPO,
    config: dict[str, Any],
    feature: str,
    mode: str,
    seed: int,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Evaluate every split and save episode and context-level tidy CSV files."""
    split_seed = int(config["environment"]["split_seed"])
    splits = build_context_splits(feature, config["environment"]["splits"], split_seed)
    episodes_per_context = int(config["evaluation"]["episodes_per_context"])
    deterministic = bool(config["evaluation"].get("deterministic", True))
    seed_offset = int(config["evaluation"]["seed_offset"])
    feature_key = carl_feature_key(feature)
    rows: list[dict[str, Any]] = []

    for split_index, (split_name, contexts) in enumerate(splits.items()):
        for context_id, context in contexts.items():
            env = make_pendulum_env(
                {context_id: context}, feature, mode, seed, static_context=True
            )
            for episode in range(episodes_per_context):
                episode_seed = (
                    seed_offset + seed + split_index * 100_000 + context_id * 1_000 + episode
                )
                observation, _ = env.reset(seed=episode_seed)
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
                        "experiment": config["experiment"]["name"],
                        "feature": feature,
                        "mode": mode,
                        "seed": seed,
                        "split": split_name,
                        "context_id": context_id,
                        "context_value": context[feature_key],
                        "episode": episode,
                        "episode_seed": episode_seed,
                        "return": episode_return,
                        "length": episode_length,
                    }
                )
            env.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "episode_returns.csv", EPISODE_FIELDS, rows)
    _write_context_summary(output_dir / "context_returns.csv", rows)
    return rows


def _write_context_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        key = tuple(row[name] for name in CONTEXT_FIELDS[:7])
        grouped[key].append(float(row["return"]))
    summaries = []
    for key, returns in grouped.items():
        summaries.append(
            dict(
                zip(
                    CONTEXT_FIELDS,
                    (*key, len(returns), statistics.fmean(returns), statistics.pstdev(returns)),
                    strict=True,
                )
            )
        )
    _write_csv(path, CONTEXT_FIELDS, summaries)


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--feature", choices=("gravity", "length", "dt"), required=True)
    parser.add_argument("--mode", choices=("hidden", "oracle"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(
        args.seed,
        deterministic_torch=bool(config["reproducibility"]["deterministic_torch"]),
    )
    model = PPO.load(args.model, device=config["training"]["device"])
    evaluate_model(model, config, args.feature, args.mode, args.seed, args.output_dir)


if __name__ == "__main__":
    main()
