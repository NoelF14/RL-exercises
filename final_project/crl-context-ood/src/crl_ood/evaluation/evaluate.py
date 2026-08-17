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
    context_normalization,
)
from crl_ood.environments.factory import make_pendulum_env, make_cartpole_env
from crl_ood.utils.metadata import load_config, load_context_manifest
from crl_ood.utils.paths import run_identifier
from crl_ood.utils.seeding import seed_everything

EPISODE_FIELDS = (
    "run_id",
    "method",
    "seed",
    "context_feature",
    "context_value",
    "split",
    "context_id",
    "episode_index",
    "episode_seed",
    "return",
    "episode_length",
    "termination_type",
)
CONTEXT_FIELDS = (
    "run_id",
    "method",
    "seed",
    "context_feature",
    "context_value",
    "split",
    "context_id",
    "episodes",
    "mean_return",
    "std_return",
)


def build_evaluation_plan(
    config: dict[str, Any],
    feature: str,
    method: str,
    seed: int,
    splits: dict[str, dict[int, dict[str, float]]],
) -> list[dict[str, Any]]:
    """Build the ordered, paired set of evaluation contexts and episode seeds."""
    episodes_per_context = int(config["evaluation"]["episodes_per_context"])
    seed_offset = int(config["evaluation"]["seed_offset"])
    feature_key = carl_feature_key(feature)
    run_id = run_identifier(config, feature, method, seed)
    plan = []
    for split_index, (split_name, contexts) in enumerate(splits.items()):
        for context_id, context in contexts.items():
            for episode_index in range(episodes_per_context):
                plan.append(
                    {
                        "run_id": run_id,
                        "method": method,
                        "seed": seed,
                        "context_feature": feature,
                        "context_value": float(context[feature_key]),
                        "split": split_name,
                        "context_id": context_id,
                        "episode_index": episode_index,
                        "episode_seed": seed_offset
                        + seed
                        + split_index * 100_000
                        + context_id * 1_000
                        + episode_index,
                    }
                )
    return plan


def evaluate_model(
    model: PPO,
    config: dict[str, Any],
    feature: str,
    method: str,
    seed: int,
    output_dir: Path,
    *,
    splits: dict[str, dict[int, dict[str, float]]] | None = None,
    normalization: tuple[float, float] | None = None,
    evaluation_plan: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate every planned episode and save episode-level and aggregate CSV."""
    if config["environment"].get("oracle_normalization") != "train_range":
        raise ValueError("Phase 0 supports only oracle_normalization: train_range")
    if splits is None:
        splits = build_context_splits(
            feature,
            config["environment"]["splits"],
            int(config["environment"]["split_seed"]),
        )
    if normalization is None:
        normalization = context_normalization(splits["train"], feature)
    if evaluation_plan is None:
        evaluation_plan = build_evaluation_plan(config, feature, method, seed, splits)

    deterministic = bool(config["evaluation"].get("deterministic", True))
    rows: list[dict[str, Any]] = []
    grouped_plan: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for planned_episode in evaluation_plan:
        grouped_plan[
            (str(planned_episode["split"]), int(planned_episode["context_id"]))
        ].append(planned_episode)

    for (split_name, context_id), episodes in grouped_plan.items():
        context = splits[split_name][context_id]
        env = make_cartpole_env( #or pendulum...
            {context_id: context},
            feature,
            method,
            seed,
            context_normalization=normalization,
            static_context=True,
        )
        for planned_episode in episodes:
            observation, _ = env.reset(seed=int(planned_episode["episode_seed"]))
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
                    **planned_episode,
                    "return": episode_return,
                    "episode_length": episode_length,
                    "termination_type": _termination_type(terminated, truncated),
                }
            )
        env.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "episode_returns.csv", EPISODE_FIELDS, rows)
    _write_context_summary(output_dir / "context_returns.csv", rows)
    return rows


def load_evaluation_plan(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    integer_fields = ("seed", "context_id", "episode_index", "episode_seed")
    for row in rows:
        for field in integer_fields:
            row[field] = int(row[field])
        row["context_value"] = float(row["context_value"])
    return rows


def _termination_type(terminated: bool, truncated: bool) -> str:
    if terminated and truncated:
        return "terminated_and_truncated"
    if terminated:
        return "terminated"
    if truncated:
        return "truncated"
    raise RuntimeError("Evaluation episode ended without termination or truncation")


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
    parser.add_argument(
        "--contexts",
        type=Path,
        help="Saved contexts.yaml; defaults to the model directory's manifest",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(
        args.seed,
        deterministic_torch=bool(config["reproducibility"]["deterministic_torch"]),
    )
    context_path = args.contexts or args.model.parent / "contexts.yaml"
    plan_path = context_path.parent / "evaluation_plan.csv"
    splits, normalization, manifest_feature = load_context_manifest(context_path)
    if manifest_feature != args.feature:
        raise ValueError(
            f"Context manifest contains {manifest_feature!r}, not {args.feature!r}"
        )
    plan = load_evaluation_plan(plan_path)
    if any(
        row["method"] != args.mode
        or row["context_feature"] != args.feature
        or row["seed"] != args.seed
        for row in plan
    ):
        raise ValueError("Saved evaluation plan does not match --mode, --feature, and --seed")
    model = PPO.load(args.model, device=config["training"]["device"])
    evaluate_model(
        model,
        config,
        args.feature,
        args.mode,
        args.seed,
        args.output_dir,
        splits=splits,
        normalization=normalization,
        evaluation_plan=plan,
    )


if __name__ == "__main__":
    main()
