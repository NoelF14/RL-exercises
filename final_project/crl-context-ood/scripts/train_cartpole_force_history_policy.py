from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from crl_ood.cartpole_encoders.wrapper import (
    make_cartpole_history_env,
)
from crl_ood.environments.context_splits import (
    build_context_splits,
    context_normalization,
)
from crl_ood.evaluation.evaluate import (
    build_evaluation_plan,
)
from crl_ood.utils.metadata import load_config
from crl_ood.utils.seeding import seed_everything


CHECKSUM = (
    "be2942887fd919057efb89525fcc8809c0d7dfb5ae6985020520cbabfd578ebf"
)


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
    raise RuntimeError(
        "episode ended without termination/truncation"
    )


def evaluate(
    model,
    *,
    config,
    splits,
    normalization,
    method,
    seed,
    checkpoint,
    output_dir,
):
    plan = build_evaluation_plan(
        config,
        "force_mag",
        method,
        seed,
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

        env = make_cartpole_history_env(
            {context_id: context},
            feature="force_mag",
            seed=seed,
            context_normalization=normalization,
            checkpoint=checkpoint,
            method=method,
            dataset_checksum=CHECKSUM,
            static_context=True,
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

            rows.append(
                {
                    **planned,
                    "return":
                        episode_return,
                    "episode_length":
                        episode_length,
                    "termination_type":
                        termination_type(
                            terminated,
                            truncated,
                        ),
                }
            )

        env.close()

    df = pd.DataFrame(rows)

    df.to_csv(
        output_dir / "episode_returns.csv",
        index=False,
    )

    context = (
        df.groupby(
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

    context.to_csv(
        output_dir / "context_returns.csv",
        index=False,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/cartpole_force_phase0.yaml"
        ),
    )

    parser.add_argument(
        "--method",
        choices=("vae", "contrastive"),
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    config = load_config(
        args.config
    )

    seed_everything(
        args.seed,
        deterministic_torch=bool(
            config["reproducibility"][
                "deterministic_torch"
            ]
        ),
    )

    splits = build_context_splits(
        "force_mag",
        config["environment"]["splits"],
        int(
            config["environment"][
                "split_seed"
            ]
        ),
        environment="cartpole",
    )

    normalization = context_normalization(
        splits["train"],
        "force_mag",
        environment="cartpole",
    )

    output_dir = (
        Path("results")
        / "cartpole_force_history_downstream"
        / args.method
        / f"seed_{args.seed}"
    )

    if output_dir.exists() and any(
        output_dir.iterdir()
    ):
        raise FileExistsError(
            f"run directory already nonempty: {output_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    base_env = make_cartpole_history_env(
        splits["train"],
        feature="force_mag",
        seed=args.seed,
        context_normalization=normalization,
        checkpoint=args.checkpoint,
        method=args.method,
        dataset_checksum=CHECKSUM,
        static_context=False,
    )

    env = Monitor(
        base_env,
        filename=str(
            output_dir / "sb3_monitor.csv"
        ),
        override_existing=True,
    )

    training = config["training"]

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=float(
            training["learning_rate"]
        ),
        n_steps=int(
            training["n_steps"]
        ),
        batch_size=int(
            training["batch_size"]
        ),
        n_epochs=int(
            training["n_epochs"]
        ),
        gamma=float(
            training["gamma"]
        ),
        gae_lambda=float(
            training["gae_lambda"]
        ),
        seed=args.seed,
        device=str(
            training["device"]
        ),
        verbose=int(
            training.get(
                "verbose",
                0,
            )
        ),
    )

    model.set_logger(
        configure(
            str(
                output_dir / "sb3_logs"
            ),
            ["csv"],
        )
    )

    model.learn(
        total_timesteps=int(
            training["total_timesteps"]
        ),
        progress_bar=False,
    )

    model.save(
        output_dir / "model"
    )

    env.close()

    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    provenance = {
        "environment":
            "cartpole",
        "context_feature":
            "force_mag",
        "method":
            args.method,
        "seed":
            args.seed,
        "encoder_checkpoint":
            str(
                args.checkpoint.resolve()
            ),
        "encoder_dataset_checksum":
            CHECKSUM,
        "source_git_commit":
            source_commit,
        "history_length":
            5,
        "latent_dim":
            8,
        "ppo_total_timesteps":
            int(
                training[
                    "total_timesteps"
                ]
            ),
    }

    (
        output_dir
        / "provenance.json"
    ).write_text(
        json.dumps(
            provenance,
            indent=2,
        )
        + "\n"
    )

    seed_everything(
        args.seed,
        deterministic_torch=bool(
            config["reproducibility"][
                "deterministic_torch"
            ]
        ),
    )

    evaluate(
        model,
        config=config,
        splits=splits,
        normalization=normalization,
        method=args.method,
        seed=args.seed,
        checkpoint=args.checkpoint,
        output_dir=output_dir,
    )

    print(output_dir)


if __name__ == "__main__":
    main()
