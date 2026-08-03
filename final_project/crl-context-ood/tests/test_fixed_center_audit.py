from __future__ import annotations

import copy
import csv
import json
import zipfile
from pathlib import Path

import yaml

from crl_ood.audit.fixed_center import build_fixed_center_matrix, inspect_fixed_center_run
from crl_ood.diagnostic.matrix import RunState
from crl_ood.environments.context_splits import build_context_splits, context_normalization
from crl_ood.utils.paths import run_directory

ROOT = Path(__file__).parents[1]
MATRIX = ROOT / "configs/audit/matrix.yaml"


def test_fixed_center_matrix_has_two_unique_oracle_outputs():
    jobs = build_fixed_center_matrix(MATRIX)
    assert len(jobs) == 2
    assert {job.seed for job in jobs} == {0, 1}
    assert {job.mode for job in jobs} == {"oracle"}
    assert {job.total_timesteps for job in jobs} == {300_000}
    assert len({job.output_dir for job in jobs}) == 2
    assert all("results/phase0_audit" in str(job.output_dir) for job in jobs)
    splits = build_context_splits("length", jobs[0].config["environment"]["splits"], seed=17)
    normalization = context_normalization(splits["train"], "length")
    assert normalization == (1.0, 1.0)
    assert (1.0 - normalization[0]) / normalization[1] == 0.0


def test_fixed_center_pending_partial_complete_detection(tmp_path):
    source = build_fixed_center_matrix(MATRIX)[0]
    config = copy.deepcopy(source.config)
    config["experiment"]["results_dir"] = str(tmp_path)
    job = copy.copy(source)
    object.__setattr__(job, "config", config)
    object.__setattr__(job, "output_dir", run_directory(config, job.feature, job.mode, job.seed))
    assert inspect_fixed_center_run(job).state is RunState.PENDING
    job.output_dir.mkdir(parents=True)
    (job.output_dir / "seed.txt").write_text("0\n", encoding="ascii")
    assert inspect_fixed_center_run(job).state is RunState.PARTIAL
    _complete(job)
    assert inspect_fixed_center_run(job).state is RunState.COMPLETE


def _complete(job):
    resolved = copy.deepcopy(job.config)
    resolved["run"] = {"context_feature": "length", "observation_mode": "oracle", "seed": job.seed}
    (job.output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    (job.output_dir / "seed.txt").write_text(f"{job.seed}\n", encoding="ascii")
    (job.output_dir / "metadata.json").write_text(json.dumps({"synthetic": True}), encoding="utf-8")
    (job.output_dir / "contexts.yaml").write_text("splits: {}\n", encoding="utf-8")
    (job.output_dir / "sb3_monitor.csv").write_text("synthetic\n", encoding="utf-8")
    (job.output_dir / "sb3_logs").mkdir(exist_ok=True)
    (job.output_dir / "sb3_logs/progress.csv").write_text("synthetic\n", encoding="utf-8")
    with zipfile.ZipFile(job.output_dir / "model.zip", "w") as archive:
        archive.writestr("synthetic", "checkpoint")
    rows = {
        "contexts.csv": {"run_id": job.job_id, "method": "oracle", "context_feature": "length", "split": "train", "context_value": 1.0},
        "evaluation_plan.csv": {"run_id": job.job_id, "method": "oracle", "seed": job.seed, "context_feature": "length", "split": "train", "episode_seed": 10000},
        "training_metrics.csv": {"run_id": job.job_id, "method": "oracle", "seed": job.seed, "context_feature": "length", "environment_steps": 200, "episode_return": -1},
        "episode_returns.csv": {"run_id": job.job_id, "method": "oracle", "seed": job.seed, "context_feature": "length", "context_value": 1.0, "split": "train", "episode_index": 0, "episode_seed": 10000, "return": -1},
        "context_returns.csv": {"run_id": job.job_id, "method": "oracle", "seed": job.seed, "context_feature": "length", "context_value": 1.0, "split": "train", "mean_return": -1},
    }
    for name, row in rows.items():
        with (job.output_dir / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=row); writer.writeheader(); writer.writerow(row)
