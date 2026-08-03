from __future__ import annotations

import copy
from pathlib import Path

from crl_ood.training.train_ppo import train_one
from crl_ood.utils.metadata import load_config


ROOT = Path(__file__).parents[1]


def test_tiny_fixed_default_context_training_smoke(tmp_path):
    config = copy.deepcopy(load_config(ROOT / "configs/diagnostic/default_100k.yaml"))
    config["experiment"]["name"] = "tiny_diagnostic_smoke"
    config["experiment"]["results_dir"] = str(tmp_path)
    config["training"].update({"total_timesteps": 8, "n_steps": 8, "batch_size": 4, "n_epochs": 1, "device": "cpu"})
    config["evaluation"]["episodes_per_context"] = 1

    run_dir = train_one(config, "length", "hidden", seed=0)

    assert run_dir == tmp_path / "tiny_diagnostic_smoke/length/hidden/seed_0"
    assert (run_dir / "model.zip").is_file()
    assert (run_dir / "episode_returns.csv").is_file()
    assert not (ROOT / "results/phase0_diagnostic").exists()
