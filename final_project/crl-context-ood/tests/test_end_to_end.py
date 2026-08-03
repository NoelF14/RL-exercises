from __future__ import annotations

import csv
import hashlib

from stable_baselines3 import PPO

from crl_ood.evaluation.evaluate import (
    evaluate_model,
    load_evaluation_plan,
)
from crl_ood.training.train_ppo import train_one
from crl_ood.utils.metadata import load_context_manifest
from crl_ood.utils.seeding import seed_everything


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_checkpoint_reload_and_deterministic_end_to_end_smoke(
    smoke_config, tmp_path
):
    config = smoke_config
    config["experiment"]["name"] = "pytest_smoke"
    config["experiment"]["results_dir"] = str(tmp_path / "results")
    config["training"].update(
        {"total_timesteps": 8, "n_steps": 8, "batch_size": 4, "n_epochs": 1}
    )
    run_dir = train_one(config, "gravity", "hidden", seed=31)

    required = {
        "resolved_config.yaml",
        "seed.txt",
        "metadata.json",
        "contexts.yaml",
        "contexts.csv",
        "evaluation_plan.csv",
        "model.zip",
        "episode_returns.csv",
        "context_returns.csv",
    }
    assert required <= {path.name for path in run_dir.iterdir()}

    splits, normalization, feature = load_context_manifest(run_dir / "contexts.yaml")
    plan = load_evaluation_plan(run_dir / "evaluation_plan.csv")
    replay_digests = []
    for replay_index in range(2):
        seed_everything(31)
        model = PPO.load(run_dir / "model.zip", device="cpu")
        replay_dir = tmp_path / f"replay_{replay_index}"
        evaluate_model(
            model,
            config,
            feature,
            "hidden",
            31,
            replay_dir,
            splits=splits,
            normalization=normalization,
            evaluation_plan=plan,
        )
        replay_digests.append(_digest(replay_dir / "episode_returns.csv"))

    assert _digest(run_dir / "episode_returns.csv") == replay_digests[0]
    assert replay_digests[0] == replay_digests[1]
    with (run_dir / "episode_returns.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert set(rows[0]) == {
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
    }
    assert {row["termination_type"] for row in rows} == {"truncated"}
