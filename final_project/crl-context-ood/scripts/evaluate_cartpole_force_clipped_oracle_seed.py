from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from crl_ood.environments.context_splits import (
    build_context_splits,
    context_normalization,
)
from crl_ood.environments.factory import make_env
from crl_ood.evaluation.evaluate import build_evaluation_plan
from crl_ood.utils.metadata import load_config
from crl_ood.utils.seeding import seed_everything


class ClippedOracleObservation(gym.ObservationWrapper):
    """Clip only the normalized scalar oracle context to [-1, 1]."""

    def __init__(self, env: gym.Env):
        super().__init__(env)

        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("oracle observation must be a Box")

        low = np.asarray(
            env.observation_space.low,
            dtype=np.float32,
        ).copy()

        high = np.asarray(
            env.observation_space.high,
            dtype=np.float32,
        ).copy()

        if low.shape != (5,):
            raise ValueError(
                f"expected 4-D state + 1-D context, got {low.shape}"
            )

        low[-1] = -1.0
        high[-1] = 1.0

        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=np.float32,
        )

    def observation(self, observation):
        obs = np.asarray(
            observation,
            dtype=np.float32,
        ).copy()

        obs[-1] = np.clip(
            obs[-1],
            -1.0,
            1.0,
        )

        return obs


def termination_type(
    terminated: bool,
    truncated: bool,
) -> str:
    if terminated and truncated:
        return "terminated_and_truncated"
    if terminated:
        return "terminated"
    if truncated:
        return "truncated"
    raise RuntimeError("episode did not end")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/cartpole_force_phase0.yaml"),
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    config = load_config(args.config)

    deterministic_torch = bool(
        config["reproducibility"]["deterministic_torch"]
    )

    seed_everything(
        args.seed,
        deterministic_torch=deterministic_torch,
    )

    splits = build_context_splits(
        "force_mag",
        config["environment"]["splits"],
        int(config["environment"]["split_seed"]),
        environment="cartpole",
    )

    normalization = context_normalization(
        splits["train"],
        "force_mag",
        environment="cartpole",
    )

    oracle_run = (
        Path("results/cartpole_force_phase0")
        / "force_mag"
        / "oracle"
        / f"seed_{args.seed}"
    )

    model_path = oracle_run / "model.zip"

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    output = (
        Path("results/cartpole_force_phase0")
        / "force_mag"
        / "oracle_clipped"
        / f"seed_{args.seed}"
    )

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"nonempty output directory: {output}"
        )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = PPO.load(
        model_path,
        device="cpu",
    )

    # Intentionally construct the RAW ORACLE evaluation plan.
    # This guarantees clipped and raw oracle use the same
    # contexts and episode seeds.
    plan = build_evaluation_plan(
        config,
        "force_mag",
        "oracle",
        args.seed,
        splits,
    )

    grouped = defaultdict(list)

    for episode in plan:
        grouped[
            (
                episode["split"],
                int(episode["context_id"]),
            )
        ].append(episode)

    rows = []

    for (
        split_name,
        context_id,
    ), episodes in grouped.items():

        context = splits[
            split_name
        ][context_id]

        base = make_env(
            "cartpole",
            {context_id: context},
            "force_mag",
            "oracle",
            args.seed,
            context_normalization=normalization,
            static_context=True,
        )

        env = ClippedOracleObservation(
            base
        )

        for planned in episodes:

            obs, _ = env.reset(
                seed=int(
                    planned["episode_seed"]
                )
            )

            terminated = False
            truncated = False
            episode_return = 0.0
            episode_length = 0

            while not (
                terminated or truncated
            ):

                action, _ = model.predict(
                    obs,
                    deterministic=True,
                )

                (
                    obs,
                    reward,
                    terminated,
                    truncated,
                    _,
                ) = env.step(action)

                episode_return += float(
                    reward
                )

                episode_length += 1

            row = dict(planned)

            row["method"] = "oracle_clipped"
            row["run_id"] = (
                f"cartpole-force_mag-"
                f"oracle_clipped-seed-{args.seed}"
            )
            row["return"] = episode_return
            row["episode_length"] = episode_length
            row["termination_type"] = (
                termination_type(
                    terminated,
                    truncated,
                )
            )

            rows.append(row)

        env.close()

    episodes = pd.DataFrame(rows)

    episodes.to_csv(
        output / "episode_returns.csv",
        index=False,
    )

    contexts = (
        episodes
        .groupby(
            [
                "run_id",
                "method",
                "seed",
                "context_feature",
                "context_value",
                "split",
                "context_id",
            ],
            as_index=False,
        )
        .agg(
            episodes=("return", "size"),
            mean_return=("return", "mean"),
            std_return=("return", "std"),
        )
    )

    contexts.to_csv(
        output / "context_returns.csv",
        index=False,
    )

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    provenance = {
        "environment": "cartpole",
        "context_feature": "force_mag",
        "seed": args.seed,
        "source_oracle_model": str(
            model_path.resolve()
        ),
        "source_git_commit": commit,
        "evaluation_plan": "raw_oracle_plan",
        "policy_context_clip": [-1.0, 1.0],
        "physical_context_unchanged": True,
    }

    (
        output
        / "diagnostic_provenance.json"
    ).write_text(
        json.dumps(
            provenance,
            indent=2,
        )
        + "\n"
    )

    print(output)


if __name__ == "__main__":
    main()
