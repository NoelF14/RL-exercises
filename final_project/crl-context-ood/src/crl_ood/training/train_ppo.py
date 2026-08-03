"""Stable-Baselines3 PPO training for Phase 0."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO

from crl_ood.environments.context_splits import build_context_splits
from crl_ood.environments.factory import make_pendulum_env
from crl_ood.evaluation.evaluate import evaluate_model
from crl_ood.utils.metadata import (
    load_config,
    resolved_run_config,
    write_run_provenance,
)
from crl_ood.utils.paths import run_directory
from crl_ood.utils.seeding import seed_everything


def train_one(config: dict[str, Any], feature: str, mode: str, seed: int) -> Path:
    """Train and evaluate one feature-mode-seed run."""
    deterministic_torch = bool(config["reproducibility"]["deterministic_torch"])
    seed_everything(seed, deterministic_torch=deterministic_torch)
    splits = build_context_splits(
        feature,
        config["environment"]["splits"],
        int(config["environment"]["split_seed"]),
    )
    run_dir = run_directory(config, feature, mode, seed)
    resolved = resolved_run_config(config, feature, mode, seed)
    write_run_provenance(run_dir, resolved, seed)

    env = make_pendulum_env(splits["train"], feature, mode, seed)
    training = config["training"]
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=float(training["learning_rate"]),
        n_steps=int(training["n_steps"]),
        batch_size=int(training["batch_size"]),
        n_epochs=int(training["n_epochs"]),
        gamma=float(training["gamma"]),
        gae_lambda=float(training["gae_lambda"]),
        seed=seed,
        device=str(training["device"]),
        verbose=int(training.get("verbose", 0)),
    )
    model.learn(total_timesteps=int(training["total_timesteps"]), progress_bar=False)
    checkpoint = run_dir / "model"
    model.save(checkpoint)
    env.close()

    seed_everything(seed, deterministic_torch=deterministic_torch)
    evaluate_model(model, config, feature, mode, seed, run_dir)
    return run_dir


def _common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main() -> None:
    parser = _common_parser(__doc__ or "Train PPO")
    parser.add_argument("--feature", choices=("gravity", "length", "dt"), required=True)
    parser.add_argument("--mode", choices=("hidden", "oracle"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    run_dir = train_one(load_config(args.config), args.feature, args.mode, args.seed)
    print(run_dir)


def phase0_main() -> None:
    parser = _common_parser("Run the Phase 0 hidden-versus-oracle matrix")
    parser.add_argument("--features", nargs="+", choices=("gravity", "length", "dt"))
    parser.add_argument("--modes", nargs="+", choices=("hidden", "oracle"))
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    features = args.features or config["experiment"]["context_features"]
    modes = args.modes or config["experiment"]["modes"]
    seeds = args.seeds or config["experiment"]["seeds"]

    for feature in features:
        for seed in seeds:
            for mode in modes:
                print(f"Training feature={feature} mode={mode} seed={seed}", flush=True)
                run_dir = train_one(config, feature, mode, int(seed))
                print(f"Saved {run_dir}", flush=True)


if __name__ == "__main__":
    main()
