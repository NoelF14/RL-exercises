from __future__ import annotations

import copy
import csv
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import yaml

from crl_ood.diagnostic.matrix import (
    RunState,
    build_diagnostic_matrix,
    inspect_run,
)
from crl_ood.utils.metadata import load_config
from crl_ood.utils.paths import run_directory


ROOT = Path(__file__).parents[1]
MATRIX = ROOT / "configs" / "diagnostic" / "matrix.yaml"


def test_complete_matrix_and_atomic_output_uniqueness():
    jobs = build_diagnostic_matrix(MATRIX)
    assert len(jobs) == 14
    assert len({job.job_id for job in jobs}) == 14
    assert len({job.output_dir for job in jobs}) == 14
    assert sum(job.experiment.startswith("default_") for job in jobs) == 4
    assert sum(job.experiment.startswith("specialist_") for job in jobs) == 6
    assert sum(job.experiment == "contextual_300k" for job in jobs) == 4
    assert {job.seed for job in jobs} == {0, 1}
    assert {job.total_timesteps for job in jobs} == {100_000, 300_000}


def test_original_phase0_paths_are_not_diagnostic_paths():
    phase0 = load_config(ROOT / "configs" / "phase0.yaml")
    original = run_directory(phase0, "length", "hidden", 0).resolve()
    assert original == (ROOT / "results/phase0/length/hidden/seed_0").resolve()
    for job in build_diagnostic_matrix(MATRIX):
        assert original != job.output_dir.resolve()
        assert (ROOT / "results/phase0").resolve() not in job.output_dir.resolve().parents


def test_dry_run_prints_complete_plan_without_creating_results(tmp_path):
    command = [
        sys.executable,
        str(ROOT / "scripts/run_phase0_diagnostic.py"),
        "--matrix-config",
        str(MATRIX),
        "--dry-run",
    ]
    completed = subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, text=True)
    assert completed.stdout.count("\n") == 16
    assert "jobs=14 concurrency=1" in completed.stdout
    assert "default_100k__length__hidden__seed_0" in completed.stdout
    assert "contextual_300k__length__oracle__seed_1" in completed.stdout
    assert not (tmp_path / "results").exists()


def test_completion_and_partial_run_detection(tmp_path):
    source = build_diagnostic_matrix(MATRIX)[0]
    config = copy.deepcopy(source.config)
    config["experiment"]["results_dir"] = str(tmp_path)
    job = copy.copy(source)
    object.__setattr__(job, "config", config)
    object.__setattr__(job, "output_dir", run_directory(config, job.feature, job.mode, job.seed))

    assert inspect_run(job).state is RunState.PENDING
    job.output_dir.mkdir(parents=True)
    (job.output_dir / "seed.txt").write_text("0\n", encoding="ascii")
    partial = inspect_run(job)
    assert partial.state is RunState.PARTIAL
    assert "missing or empty artifacts" in partial.detail

    _write_complete_artifacts(job)
    assert inspect_run(job).state is RunState.COMPLETE
    (job.output_dir / "seed.txt").write_text("999\n", encoding="ascii")
    mismatch = inspect_run(job)
    assert mismatch.state is RunState.PARTIAL
    assert "planned seed" in mismatch.detail


def _write_complete_artifacts(job):
    run_dir = job.output_dir
    resolved = copy.deepcopy(job.config)
    resolved["run"] = {
        "context_feature": job.feature,
        "observation_mode": job.mode,
        "seed": job.seed,
    }
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    (run_dir / "seed.txt").write_text(f"{job.seed}\n", encoding="ascii")
    (run_dir / "metadata.json").write_text(json.dumps({"synthetic": True}), encoding="utf-8")
    (run_dir / "contexts.yaml").write_text("splits: {}\n", encoding="utf-8")
    (run_dir / "sb3_monitor.csv").write_text("synthetic\n", encoding="utf-8")
    (run_dir / "sb3_logs").mkdir(exist_ok=True)
    (run_dir / "sb3_logs/progress.csv").write_text("synthetic\n", encoding="utf-8")
    with zipfile.ZipFile(run_dir / "model.zip", "w") as archive:
        archive.writestr("synthetic", "checkpoint")
    rows = {
        "contexts.csv": {
            "run_id": job.job_id, "method": job.mode, "context_feature": "length",
            "split": "train", "context_value": 1.0,
        },
        "evaluation_plan.csv": {
            "run_id": job.job_id, "method": job.mode, "seed": job.seed,
            "context_feature": "length", "split": "train", "episode_seed": 10_000,
        },
        "training_metrics.csv": {
            "run_id": job.job_id, "method": job.mode, "seed": job.seed,
            "context_feature": "length", "environment_steps": 200, "episode_return": -1.0,
        },
        "episode_returns.csv": {
            "run_id": job.job_id, "method": job.mode, "seed": job.seed,
            "context_feature": "length", "context_value": 1.0, "split": "train",
            "episode_index": 0, "episode_seed": 10_000, "return": -1.0,
        },
        "context_returns.csv": {
            "run_id": job.job_id, "method": job.mode, "seed": job.seed,
            "context_feature": "length", "context_value": 1.0, "split": "train",
            "mean_return": -1.0,
        },
    }
    for filename, row in rows.items():
        with (run_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=row)
            writer.writeheader()
            writer.writerow(row)
