from __future__ import annotations

import copy
from pathlib import Path

from crl_ood.goal_pilot.matrix import RunState, build_goal_pilot_matrix, inspect_run
from crl_ood.goal_pilot.run import train_one

ROOT = Path(__file__).parents[1]


def test_tiny_budget_goal_training_and_full_split_evaluation(tmp_path):
    source = build_goal_pilot_matrix(ROOT / "configs/goal_pilot/matrix.yaml")[0]
    config = copy.deepcopy(source.config)
    config["training"].update(
        {"total_timesteps": 256, "n_steps": 64, "batch_size": 32, "n_epochs": 1, "device": "cpu"}
    )
    config["evaluation"]["episodes_per_context"] = 1
    job = copy.copy(source)
    object.__setattr__(job, "config", config)
    object.__setattr__(job, "total_timesteps", 256)
    object.__setattr__(job, "output_dir", tmp_path / source.job_id)

    run_dir = train_one(job)

    assert run_dir == job.output_dir
    assert inspect_run(job).state is RunState.COMPLETE
    assert (run_dir / "model.zip").is_file()
    assert (run_dir / "sb3_logs/progress.csv").is_file()
    assert (run_dir / "episode_returns.csv").read_text(encoding="utf-8").count("\n") == 14
    context_text = (run_dir / "contexts.yaml").read_text(encoding="utf-8")
    assert "ood_left:" in context_text and "ood_right:" in context_text
