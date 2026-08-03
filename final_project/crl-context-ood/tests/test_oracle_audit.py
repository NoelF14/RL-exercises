from __future__ import annotations

import copy
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from crl_ood.audit.evaluate import (
    ABLATION_MODES,
    _rollout,
    build_oracle_jobs,
    build_shuffled_mapping,
    build_specialist_jobs,
    load_audit_config,
    modify_oracle_observation,
    resolve_checkpoint,
    run_audit_task,
)
from crl_ood.utils.metadata import load_context_manifest

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/audit/oracle_audit.yaml"


def test_specialist_checkpoint_cross_product_and_path_resolution():
    config = load_audit_config(CONFIG)
    jobs = build_specialist_jobs(config)
    assert {(job.label, job.seed) for job in jobs} == {
        (label, seed) for label in ("low", "center", "high") for seed in (0, 1)
    }
    assert len({job.output_dir for job in jobs}) == 6
    for job in jobs:
        assert job.checkpoint == resolve_checkpoint(
            ROOT / "results/phase0_diagnostic", f"specialist_{job.label}_300k", "hidden", job.seed
        )
        with (job.source_run_dir / "evaluation_plan.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        contexts = {(row["split"], row["context_id"]) for row in rows}
        assert len(contexts) == 27
        assert {row["split"] for row in rows} == {"train", "id_test", "ood_low", "ood_high"}
    assert len(jobs) * 27 == 162


def test_specialist_and_oracle_plans_preserve_exact_paired_episode_seeds():
    config = load_audit_config(CONFIG)
    for seed in (0, 1):
        hidden_plans = []
        for job in [job for job in build_specialist_jobs(config) if job.seed == seed]:
            hidden_plans.append(_paired_fields(job.source_run_dir / "evaluation_plan.csv"))
        assert hidden_plans[0] == hidden_plans[1] == hidden_plans[2]
        oracle = next(job for job in build_oracle_jobs(config) if job.seed == seed)
        assert hidden_plans[0] == _paired_fields(oracle.source_run_dir / "evaluation_plan.csv")


def test_shuffled_mapping_is_deterministic_deranged_and_multiset_preserving():
    job = build_oracle_jobs(load_audit_config(CONFIG))[0]
    _, normalization, _ = load_context_manifest(job.source_run_dir / "contexts.yaml")
    plan = _read_plan(job.source_run_dir / "evaluation_plan.csv")
    first = build_shuffled_mapping(plan, normalization, seed=0)
    second = build_shuffled_mapping(plan, normalization, seed=0)
    assert first == second
    assert all(row["true_normalized_context"] != row["shuffled_normalized_context"] for row in first)
    assert sorted(row["true_normalized_context"] for row in first) == sorted(
        row["shuffled_normalized_context"] for row in first
    )


@pytest.mark.parametrize("mode", ABLATION_MODES)
def test_observation_ablation_changes_only_oracle_scalar(mode):
    observation = np.array([1.0, 2.0, 3.0, 0.75], dtype=np.float32)
    changed = modify_oracle_observation(observation, mode, shuffled_scalar=-0.25)
    np.testing.assert_array_equal(changed[:3], observation[:3])
    np.testing.assert_array_equal(observation, np.array([1.0, 2.0, 3.0, 0.75], dtype=np.float32))
    expected = {"true_context": 0.75, "zero_context": 0.0, "shuffled_context": -0.25}[mode]
    assert changed[-1] == pytest.approx(expected)


class _ZeroPolicy:
    def __init__(self):
        self.observations = []

    def predict(self, observation, deterministic=True):
        self.observations.append(np.asarray(observation).copy())
        return np.array([0.0], dtype=np.float32), None


@pytest.mark.parametrize(("mode", "expected_scalar"), [("true_context", 0.0), ("zero_context", 0.0)])
def test_tiny_oracle_evaluation_keeps_actual_environment_context(mode, expected_scalar):
    splits = {"train": {0: {"l": 1.0}}}
    plan = [{
        "run_id": "tiny", "method": "oracle", "seed": 0, "context_feature": "length",
        "context_value": 1.0, "split": "train", "context_id": 0,
        "episode_index": 0, "episode_seed": 123,
    }]
    model = _ZeroPolicy()
    rows = _rollout(
        model, splits, (1.0, 1.0), plan, "length", "oracle", 0,
        observation_mode=mode,
    )
    assert len(rows) == 1 and rows[0]["episode_seed"] == 123
    assert model.observations
    assert all(observation[-1] == pytest.approx(expected_scalar) for observation in model.observations)


def test_audit_refuses_existing_output_by_default(tmp_path):
    config = copy.deepcopy(load_audit_config(CONFIG))
    config["audit"]["results_dir"] = str(tmp_path)
    existing = tmp_path / "specialist_transfer"
    existing.mkdir(parents=True)
    (existing / "marker.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError, match="--resume"):
        run_audit_task("specialist-transfer", config)
    assert (existing / "marker.txt").read_text(encoding="utf-8") == "preserve"


def _paired_fields(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            (row["seed"], row["context_value"], row["split"], row["context_id"], row["episode_index"], row["episode_seed"])
            for row in csv.DictReader(handle)
        ]


def _read_plan(path):
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in ("seed", "context_id", "episode_index", "episode_seed"):
            row[field] = int(row[field])
        row["context_value"] = float(row["context_value"])
    return rows


def test_result_only_analyzer_imports_no_rl_stack():
    code = (
        "import sys; import crl_ood.analysis.analyze_oracle_audit; "
        "blocked=('carl','gym','gymnasium','stable_baselines3','torch'); "
        "assert not any(any(n == b or n.startswith(b + '.') for b in blocked) for n in sys.modules), "
        "sorted(n for n in sys.modules if any(n == b or n.startswith(b + '.') for b in blocked))"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_completed_results_and_phase0_config_match_git_byte_for_byte():
    tracked = subprocess.run(
        ["git", "ls-files", "-z", "configs/phase0.yaml", "results/phase0", "results/phase0_diagnostic"],
        cwd=ROOT, check=True, capture_output=True,
    ).stdout.split(b"\0")
    paths = [Path(value.decode()) for value in tracked if value]
    assert paths
    for relative in paths:
        committed = subprocess.run(
            ["git", "show", f"HEAD:./{relative.as_posix()}"], cwd=ROOT, check=True, capture_output=True
        ).stdout
        assert (ROOT / relative).read_bytes() == committed
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z", "results/phase0", "results/phase0_diagnostic"],
        cwd=ROOT, check=True, capture_output=True,
    ).stdout
    assert untracked == b""
