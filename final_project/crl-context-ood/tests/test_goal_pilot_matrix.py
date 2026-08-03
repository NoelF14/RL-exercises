from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import pytest

from crl_ood.goal_pilot.matrix import RunState, build_goal_pilot_matrix, inspect_run
from crl_ood.goal_pilot.run import build_evaluation_plan
from crl_ood.goal_pilot.spec import EXPECTED_SPLITS, goal_normalization, goal_splits

ROOT = Path(__file__).parents[1]
MATRIX = ROOT / "configs/goal_pilot/matrix.yaml"


def test_exact_disjoint_goal_splits_and_oracle_normalization():
    jobs = build_goal_pilot_matrix(MATRIX)
    splits = goal_splits(jobs[0].config)
    assert {name: tuple(values.values()) for name, values in splits.items()} == EXPECTED_SPLITS
    flattened = [value for split in splits.values() for value in split.values()]
    assert len(flattened) == len(set(flattened))
    assert goal_normalization(splits["train"]) == (0.0, 0.6)


def test_complete_unique_matrix_and_specialist_center_deduplication():
    jobs = build_goal_pilot_matrix(MATRIX)
    assert len(jobs) == len({job.job_id for job in jobs}) == len({job.output_dir for job in jobs}) == 12
    assert all(job.total_timesteps == 300_000 for job in jobs)
    assert sum(job.kind == "contextual" for job in jobs) == 4
    assert sum(job.kind.startswith("specialist_") for job in jobs) == 6
    assert sum(job.kind == "fixed_center" for job in jobs) == 2
    reused = [job for job in jobs if "fixed_center_hidden" in job.roles]
    assert len(reused) == 2
    assert {job.kind for job in reused} == {"specialist_center"}
    assert not any(job.kind == "fixed_center" and job.mode == "hidden" for job in jobs)


def test_evaluation_seeds_are_deterministic_and_paired_across_methods():
    jobs = build_goal_pilot_matrix(MATRIX)
    hidden = next(job for job in jobs if job.kind == "contextual" and job.mode == "hidden" and job.seed == 0)
    oracle = next(job for job in jobs if job.kind == "contextual" and job.mode == "oracle" and job.seed == 0)
    hidden_plan = build_evaluation_plan(hidden)
    oracle_plan = build_evaluation_plan(oracle)
    paired_fields = ("seed", "split", "context_id", "target_angle", "episode_index", "episode_seed")
    assert [tuple(row[field] for field in paired_fields) for row in hidden_plan] == [tuple(row[field] for field in paired_fields) for row in oracle_plan]
    assert hidden_plan == build_evaluation_plan(hidden)
    assert len({row["episode_seed"] for row in hidden_plan}) == len(hidden_plan)


def test_dry_run_is_complete_and_does_not_import_rl_stack(tmp_path):
    code = (
        "import sys; blocked={'carl','gym','gymnasium','stable_baselines3','torch'}; "
        "sys.meta_path.insert(0,type('B',(),{'find_spec':lambda s,n,p=None,t=None: "
        "(_ for _ in ()).throw(RuntimeError(n)) if n.split('.')[0] in blocked else None})()); "
        "from crl_ood.goal_pilot.run import main; "
        f"sys.argv=['run','--matrix-config',{str(MATRIX)!r},'--dry-run']; main()"
    )
    completed = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True, capture_output=True, text=True)
    assert "jobs=12 unique_atomic_runs=12 concurrency=1" in completed.stdout
    assert completed.stdout.count("300000") == 12
    assert not (tmp_path / "results").exists()


def test_partial_and_overwrite_protection(tmp_path):
    source = build_goal_pilot_matrix(MATRIX)[0]
    job = copy.copy(source)
    object.__setattr__(job, "output_dir", tmp_path / source.job_id)
    assert inspect_run(job).state is RunState.PENDING
    job.output_dir.mkdir()
    (job.output_dir / "seed.txt").write_text("0\n", encoding="ascii")
    partial = inspect_run(job)
    assert partial.state is RunState.PARTIAL
    assert "missing or empty artifacts" in partial.detail
    # The training entry point refuses before touching the existing sentinel.
    from crl_ood.goal_pilot.run import train_one
    with pytest.raises(FileExistsError):
        train_one(job)
    assert (job.output_dir / "seed.txt").read_text(encoding="ascii") == "0\n"


def test_protected_namespaces_are_never_output_targets():
    protected = [(ROOT / name).resolve() for name in ("results/phase0", "results/phase0_diagnostic", "results/phase0_audit")]
    for job in build_goal_pilot_matrix(MATRIX):
        path = job.output_dir.resolve()
        assert all(root != path and root not in path.parents for root in protected)
