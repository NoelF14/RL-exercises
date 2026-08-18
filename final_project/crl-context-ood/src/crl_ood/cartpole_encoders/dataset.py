"""Matched context-independent CartPole force-magnitude trajectory dataset."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from crl_ood.environments.context_splits import (
    build_context_splits,
    context_normalization,
)
from crl_ood.environments.factory import make_env
from crl_ood.pointrobot_encoders.dataset import (
    dataset_checksum,
    save_dataset,
    transition_features,
)
from crl_ood.utils.metadata import load_config
from crl_ood.utils.paths import project_root


def matched_actions(
    trajectory_seed: int,
    horizon: int,
) -> np.ndarray:
    """Context-independent Bernoulli action sequence shared across all forces."""
    rng = np.random.default_rng(int(trajectory_seed))
    actions = rng.integers(
        0,
        2,
        size=horizon,
        dtype=np.int64,
    )
    return actions.astype(np.float32)[:, None]


def _rollout(
    context: dict[str, float],
    actions: np.ndarray,
    trajectory_seed: int,
    normalization: tuple[float, float],
) -> dict[str, np.ndarray] | None:
    env = make_env(
        "cartpole",
        {0: context},
        feature="force_mag",
        mode="hidden",
        seed=trajectory_seed,
        context_normalization=normalization,
        static_context=True,
    )

    state, _ = env.reset(seed=int(trajectory_seed))

    states = [np.asarray(state, dtype=np.float32)]
    rewards = []
    next_states = []
    terminated = []
    truncated = []

    survived = True

    for action_row in actions:
        action = int(action_row[0])

        nxt, reward, term, trunc, _ = env.step(action)

        nxt = np.asarray(nxt, dtype=np.float32)

        rewards.append(float(reward))
        next_states.append(nxt)
        terminated.append(bool(term))
        truncated.append(bool(trunc))
        states.append(nxt)

        if term or trunc:
            survived = False
            break

    env.close()

    if not survived or len(rewards) != len(actions):
        return None

    return {
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "next_states": np.asarray(next_states, dtype=np.float32),
        "terminated": np.asarray(terminated, dtype=np.bool_),
        "truncated": np.asarray(truncated, dtype=np.bool_),
    }


def collect_dataset(
    phase0_config: str | Path,
    *,
    horizon: int = 20,
    trajectories_per_context: int = 200,
    candidate_seed_offset: int = 30_000,
    validation_fraction: float = 0.20,
    split_seed: int = 31,
    max_candidates: int = 10_000,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if horizon < 10:
        raise ValueError("horizon must permit H=5 plus K=5")

    if trajectories_per_context < 1:
        raise ValueError("trajectories_per_context must be positive")

    config = load_config(Path(phase0_config))

    splits = build_context_splits(
        "force_mag",
        config["environment"]["splits"],
        int(config["environment"]["split_seed"]),
        environment="cartpole",
    )

    train_context_mapping = splits["train"]

    normalization = context_normalization(
        train_context_mapping,
        "force_mag",
        environment="cartpole",
    )

    # Stable ordering by actual physical force rather than shuffled CARL context id.
    contexts = sorted(
        (
            {
                key: float(value)
                for key, value in context.items()
            }
            for context in train_context_mapping.values()
        ),
        key=lambda context: context["force_mag"],
    )

    accepted: list[
        tuple[int, np.ndarray, list[dict[str, np.ndarray]]]
    ] = []

    candidates_tested = 0

    for candidate_index in range(max_candidates):
        if len(accepted) >= trajectories_per_context:
            break

        trajectory_seed = candidate_seed_offset + candidate_index
        candidates_tested += 1

        actions = matched_actions(
            trajectory_seed,
            horizon,
        )

        rollouts = []
        valid = True

        for context in contexts:
            rollout = _rollout(
                context,
                actions,
                trajectory_seed,
                normalization,
            )

            if rollout is None:
                valid = False
                break

            rollouts.append(rollout)

        if valid:
            accepted.append(
                (
                    trajectory_seed,
                    actions,
                    rollouts,
                )
            )

    if len(accepted) != trajectories_per_context:
        raise RuntimeError(
            "Could not obtain requested matched trajectories: "
            f"{len(accepted)}/{trajectories_per_context} "
            f"after {candidates_tested} candidates"
        )

    split_rng = np.random.default_rng(int(split_seed))

    validation_n = max(
        1,
        int(
            round(
                trajectories_per_context
                * float(validation_fraction)
            )
        ),
    )

    validation_indices = set(
        split_rng.choice(
            trajectories_per_context,
            size=validation_n,
            replace=False,
        ).tolist()
    )

    records: dict[str, list[Any]] = {
        key: []
        for key in (
            "states",
            "actions",
            "rewards",
            "next_states",
            "terminated",
            "truncated",
            "contexts",
            "timesteps",
            "trajectory_seeds",
            "episode_ids",
            "assignments",
        )
    }

    for context_index, context in enumerate(contexts):
        force = float(context["force_mag"])

        for trajectory_index, (
            trajectory_seed,
            actions,
            rollouts,
        ) in enumerate(accepted):
            rollout = rollouts[context_index]

            # Defensive matching check.
            if not np.array_equal(
                rollout["actions"],
                actions,
            ):
                raise RuntimeError(
                    "matched action sequence changed across contexts"
                )

            episode_id = (
                context_index * trajectories_per_context
                + trajectory_index
            )

            records["states"].append(rollout["states"])
            records["actions"].append(rollout["actions"])
            records["rewards"].append(rollout["rewards"])
            records["next_states"].append(
                rollout["next_states"]
            )
            records["terminated"].append(
                rollout["terminated"]
            )
            records["truncated"].append(
                rollout["truncated"]
            )
            records["contexts"].append(force)
            records["timesteps"].append(
                np.arange(
                    horizon,
                    dtype=np.int32,
                )
            )
            records["trajectory_seeds"].append(
                trajectory_seed
            )
            records["episode_ids"].append(
                episode_id
            )
            records["assignments"].append(
                1
                if trajectory_index in validation_indices
                else 0
            )

    arrays = {
        "states": np.asarray(
            records["states"],
            dtype=np.float32,
        ),
        "actions": np.asarray(
            records["actions"],
            dtype=np.float32,
        ),
        "rewards": np.asarray(
            records["rewards"],
            dtype=np.float32,
        ),
        "next_states": np.asarray(
            records["next_states"],
            dtype=np.float32,
        ),
        "terminated": np.asarray(
            records["terminated"],
            dtype=np.bool_,
        ),
        "truncated": np.asarray(
            records["truncated"],
            dtype=np.bool_,
        ),
        "contexts": np.asarray(
            records["contexts"],
            dtype=np.float32,
        ),
        "timesteps": np.asarray(
            records["timesteps"],
            dtype=np.int32,
        ),
        "trajectory_seeds": np.asarray(
            records["trajectory_seeds"],
            dtype=np.int64,
        ),
        "episode_ids": np.asarray(
            records["episode_ids"],
            dtype=np.int64,
        ),
        "assignments": np.asarray(
            records["assignments"],
            dtype=np.int8,
        ),
    }

    features = transition_features(
        arrays["states"][:, :-1],
        arrays["actions"],
        arrays["rewards"],
        arrays["next_states"],
    )

    if features.shape[-1] != 10:
        raise RuntimeError(
            f"expected CartPole transition_dim=10, got {features.shape[-1]}"
        )

    fit = (
        features[arrays["assignments"] == 0]
        .reshape(-1, features.shape[-1])
        .astype(np.float64)
    )

    mean = fit.mean(axis=0)
    std = fit.std(axis=0)
    std[std < 1e-8] = 1.0

    arrays["normalization_mean"] = mean.astype(
        np.float32
    )
    arrays["normalization_std"] = std.astype(
        np.float32
    )

    accepted_seeds = [
        int(item[0])
        for item in accepted
    ]

    metadata: dict[str, Any] = {
        "format_version": 1,
        "environment": "CARLCartPole",
        "context_feature": "force_mag",
        "immutable": True,
        "source_git_commit": _commit(),
        "source_phase0_config": str(
            Path(phase0_config)
        ),
        "contexts": [
            float(context["force_mag"])
            for context in contexts
        ],
        "horizon": int(horizon),
        "behavior_policy": {
            "name": "matched_open_loop_bernoulli",
            "distribution": "iid Bernoulli(0.5) over actions {0,1}",
            "context_independent": True,
            "same_seed_and_actions_across_contexts": True,
            "candidate_seed_offset": int(
                candidate_seed_offset
            ),
            "survival_filter": (
                "retain trajectory seed only if the identical "
                f"{horizon}-step action sequence survives all "
                "training force contexts"
            ),
            "candidates_tested": int(
                candidates_tested
            ),
            "accepted": int(
                trajectories_per_context
            ),
            "acceptance_fraction": float(
                trajectories_per_context
                / candidates_tested
            ),
        },
        "trajectory_seeds": accepted_seeds,
        "validation": {
            "fraction": float(
                validation_fraction
            ),
            "split_seed": int(split_seed),
            "trajectory_indices": sorted(
                validation_indices
            ),
            "assignment_shared_across_contexts": True,
        },
        "assignment_encoding": {
            "0": "train",
            "1": "validation",
        },
        "episode_count": int(
            len(arrays["episode_ids"])
        ),
        "transition_count": int(
            len(arrays["episode_ids"])
            * horizon
        ),
        "array_schema": {
            key: {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
            }
            for key, value in arrays.items()
        },
        "normalization": {
            "fit_scope": (
                "training assignment only, "
                "training force contexts only"
            ),
            "mean": arrays[
                "normalization_mean"
            ].tolist(),
            "std": arrays[
                "normalization_std"
            ].tolist(),
        },
    }

    metadata["dataset_checksum"] = dataset_checksum(
        arrays,
        metadata,
    )

    return arrays, metadata


def _commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root(),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_and_save(
    phase0_config: str | Path,
    output_dir: str | Path,
) -> Path:
    arrays, metadata = collect_dataset(
        phase0_config,
    )

    return save_dataset(
        output_dir,
        arrays,
        metadata,
    )
