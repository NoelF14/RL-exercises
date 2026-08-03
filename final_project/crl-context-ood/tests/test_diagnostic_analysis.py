from __future__ import annotations

import copy
import csv
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from crl_ood.analysis.analyze_diagnostic import analyze_diagnostic
from crl_ood.diagnostic.matrix import build_diagnostic_matrix


ROOT = Path(__file__).parents[1]
MATRIX = ROOT / "configs/diagnostic/matrix.yaml"


def test_diagnostic_analyzer_import_does_not_load_carl():
    command = (
        "import sys; import crl_ood.analysis.analyze_diagnostic; "
        "assert not any(name == 'carl' or name.startswith('carl.') for name in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", command], check=True)


def test_result_only_diagnostic_analysis_with_synthetic_matrix(tmp_path):
    results_root = tmp_path / "phase0_diagnostic"
    for job in build_diagnostic_matrix(MATRIX):
        _write_synthetic_job(results_root, job)

    paths = analyze_diagnostic(results_root, tmp_path / "analysis")
    seed_results = pd.read_csv(paths["seed_results"])
    paired = pd.read_csv(paths["default_paired"])
    contextual = pd.read_csv(paths["contextual_comparison"])
    ood = pd.read_csv(paths["ood_seed"])

    assert len(seed_results) == 26
    assert paired["delta_300k_minus_100k"].tolist() == pytest.approx([30.0, 30.0])
    assert contextual[contextual["split"] == "train"]["oracle_minus_hidden"].tolist() == pytest.approx([10.0, 10.0])
    assert contextual[contextual["split"] == "id_test"]["oracle_minus_hidden"].tolist() == pytest.approx([15.0, 15.0])
    assert set(ood["analysis_role"]) == {"descriptive_only"}
    assert not ood["eligible_for_tuning_or_selection"].any()
    for path in paths.values():
        assert path.is_file() and path.stat().st_size > 0
    for path in paths.values():
        if path.suffix == ".csv":
            assert "confidence" not in " ".join(pd.read_csv(path).columns).lower()


def _write_synthetic_job(results_root, job):
    run_dir = results_root / job.experiment / "length" / job.mode / f"seed_{job.seed}"
    run_dir.mkdir(parents=True)
    resolved = copy.deepcopy(job.config)
    resolved["run"] = {
        "context_feature": "length",
        "observation_mode": job.mode,
        "seed": job.seed,
    }
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    fields = (
        "run_id", "method", "seed", "context_feature", "context_value", "split",
        "context_id", "episode_index", "episode_seed", "return", "episode_length",
        "termination_type",
    )
    splits = ("train", "id_test", "ood_low", "ood_high") if job.experiment == "contextual_300k" else ("train",)
    with (run_dir / "episode_returns.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        counts = {"train": 9, "id_test": 8, "ood_low": 5, "ood_high": 5}
        for split_index, split in enumerate(splits):
            count = counts[split] if job.experiment == "contextual_300k" else 1
            for context_id in range(count):
                for episode_index, offset in enumerate((-1.0, 1.0)):
                    value = _synthetic_return(job, split) + offset
                    writer.writerow(
                        {
                            "run_id": job.job_id, "method": job.mode, "seed": job.seed,
                            "context_feature": "length",
                            "context_value": _context_value(job) + context_id * 0.001,
                            "split": split, "context_id": context_id,
                            "episode_index": episode_index,
                            "episode_seed": 10_000 + split_index * 10_000 + context_id * 100 + episode_index,
                            "return": value, "episode_length": 200,
                            "termination_type": "truncated",
                        }
                    )


def _context_value(job):
    if "low" in job.experiment:
        return 0.8
    if "high" in job.experiment:
        return 1.2
    return 1.0


def _synthetic_return(job, split):
    if job.experiment == "default_100k":
        return -100.0 + job.seed
    if job.experiment == "default_300k":
        return -70.0 + job.seed
    if job.experiment.startswith("specialist_"):
        return {"low": -90.0, "center": -80.0, "high": -70.0}[job.experiment.split("_")[1]] + job.seed
    hidden = {"train": -100.0, "id_test": -110.0, "ood_low": -150.0, "ood_high": -140.0}[split] + job.seed
    gap = {"train": 10.0, "id_test": 15.0, "ood_low": 20.0, "ood_high": 25.0}[split]
    return hidden if job.mode == "hidden" else hidden + gap
