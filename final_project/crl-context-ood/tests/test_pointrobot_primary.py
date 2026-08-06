from __future__ import annotations

import ast
import csv
import hashlib
import json
import shutil
from pathlib import Path

import pytest
import yaml

from crl_ood.analysis.analyze_pointrobot_primary import analyze, paired_bootstrap_interval
from crl_ood.pointrobot_primary.spec import (METHODS, SEEDS, SPLITS, build_downstream_jobs,
    build_encoder_jobs, load_spec, validate_encoder_runs, validate_timestep_budget)

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "configs/pointrobot_primary/spec.yaml"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[dict, Path]:
    spec = load_spec(SPEC_PATH)
    spec["experiment"]["results_dir"] = "results/pointrobot_primary"
    checksum = "a" * 64
    dataset = tmp_path / "results/pointrobot_primary/dataset/full"
    dataset.mkdir(parents=True)
    (dataset / "dataset.sha256").write_text(checksum + "\n")
    (dataset / "dataset.json").write_text(json.dumps({"dataset_checksum": checksum}))
    for method in ("vae", "contrastive"):
        for seed in SEEDS:
            run = tmp_path / "results/pointrobot_primary/encoders" / method / f"seed_{seed}"
            run.mkdir(parents=True)
            checkpoint = run / "best.pt"; checkpoint.write_bytes(f"{method}-{seed}".encode())
            (run / "provenance.json").write_text(json.dumps({"method": method, "seed": seed,
                "dataset_checksum": checksum, "source_commit": spec["experiment"]["source_commit"],
                "parameter_counts": {"downstream_retained": 14536}}))
            (run / "checkpoint_selection.json").write_text(json.dumps({
                "selection_scope": "held-out training-context trajectories only", "ood_used": False}))
            (run / "checkpoint_manifest.json").write_text(json.dumps({"best.pt": _sha(checkpoint)}))
            (run / "run.log").write_text(f"COMPLETE method={method} seed={seed} updates=20000\n")
    return spec, tmp_path


def _downstream_fixture(tmp_path: Path) -> tuple[dict, Path]:
    spec, root = _fixture(tmp_path)
    jobs = build_downstream_jobs(spec, root)
    for job in jobs:
        job.output_dir.mkdir(parents=True)
        provenance = {"method": job.method, "encoder_seed": job.encoder_seed, "policy_seed": job.policy_seed,
            "requested_timesteps": 200000, "actual_complete_rollout_timesteps": 200704,
            "rollout_quantum": 2048, "dataset_checksum": job.dataset_checksum,
            "encoder_checkpoint_path": str(job.checkpoint) if job.checkpoint else None,
            "encoder_checkpoint_sha256": job.checkpoint_sha256, "source_commit": spec["experiment"]["source_commit"],
            "configuration_checksum": spec["_configuration_checksum"], "selection_split": "training contexts only",
            "ood_role": "descriptive/scientific evaluation only"}
        (job.output_dir / "provenance.json").write_text(json.dumps(provenance))
        (job.output_dir / "run.log").write_text("COMPLETE\n")
        context_rows = []
        for split, angles in SPLITS.items():
            for angle in angles:
                advantage = {"no_context": 0.0, "oracle": 5.0, "vae": 3.0, "contrastive": 2.5}[job.method]
                context_rows.append({"method": job.method, "policy_seed": job.policy_seed,
                    "encoder_seed": "" if job.encoder_seed is None else job.encoder_seed, "split": split,
                    "goal_angle": angle, "mean_return": -20 + advantage + job.policy_seed * .1 - abs(angle),
                    "return_std": 1.0, "success_rate": advantage / 5, "mean_final_distance": 1 - advantage / 10,
                    "mean_minimum_distance": .8 - advantage / 12, "mean_first_success_timestep": 20,
                    "evaluation_episode_count": 10})
        with (job.output_dir / "context_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(context_rows[0])); writer.writeheader(); writer.writerows(context_rows)
        with (job.output_dir / "training_progress.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("timesteps", "mean_recent_episode_return", "recent_episode_count"))
            writer.writeheader(); writer.writerow({"timesteps": 2048, "mean_recent_episode_return": -20, "recent_episode_count": 10})
    return spec, root


def test_frozen_spec_and_exactly_ten_encoder_jobs(tmp_path):
    spec, root = _fixture(tmp_path)
    jobs = build_encoder_jobs(spec, root)
    assert len(jobs) == 10
    assert {(job.method, job.encoder_seed) for job in jobs} == {(method, seed) for method in ("vae", "contrastive") for seed in SEEDS}
    assert len({job.dataset_checksum for job in jobs}) == 1


def test_exactly_twenty_downstream_jobs_and_seed_pairing(tmp_path):
    spec, root = _fixture(tmp_path)
    jobs = build_downstream_jobs(spec, root)
    assert len(jobs) == 20 and len({(job.method, job.policy_seed) for job in jobs}) == 20
    assert all(job.encoder_seed == job.policy_seed for job in jobs if job.method in ("vae", "contrastive"))
    assert all({job.policy_seed for job in jobs if job.method == method} == set(SEEDS) for method in METHODS)


def test_refuses_one_checkpoint_for_all_seeds(tmp_path):
    spec, root = _fixture(tmp_path)
    shared = root / "shared.pt"; shared.write_text("shared")
    shared_hash = _sha(shared)
    for seed in SEEDS:
        run = root / "results/pointrobot_primary/encoders/vae" / f"seed_{seed}"
        (run / "best.pt").unlink(); (run / "best.pt").symlink_to(shared)
        (run / "checkpoint_manifest.json").write_text(json.dumps({"best.pt": shared_hash}))
    with pytest.raises(ValueError, match="one-checkpoint"): build_downstream_jobs(spec, root)


def test_unavailable_matrix_with_missing_encoder_run(tmp_path):
    spec, root = _fixture(tmp_path)
    (root / "results/pointrobot_primary/encoders/vae/seed_4/run.log").unlink()
    with pytest.raises(RuntimeError, match="unavailable"): build_downstream_jobs(spec, root)


def test_dataset_checksum_equality_and_checkpoint_hash_verification(tmp_path):
    spec, root = _fixture(tmp_path)
    run = root / "results/pointrobot_primary/encoders/vae/seed_3"
    provenance = json.loads((run / "provenance.json").read_text()); provenance["dataset_checksum"] = "b" * 64
    (run / "provenance.json").write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="dataset"): validate_encoder_runs(spec, root)
    provenance["dataset_checksum"] = "a" * 64; (run / "provenance.json").write_text(json.dumps(provenance))
    (run / "best.pt").write_text("tampered")
    with pytest.raises(ValueError, match="hash"): validate_encoder_runs(spec, root)


def test_requested_vs_complete_rollout_timestep_validation():
    validate_timestep_budget(200000, 200704, 2048)
    with pytest.raises(ValueError): validate_timestep_budget(200000, 200000, 2048)


def test_context_split_near_far_and_seed_only_analysis(tmp_path):
    spec, root = _downstream_fixture(tmp_path)
    output = analyze(root / "results/pointrobot_primary", SPEC_PATH)
    contexts = list(csv.DictReader((output / "primary_context_results.csv").open()))
    assert len(contexts) == 20 * 13
    assert {(row["split"], float(row["goal_angle"])) for row in contexts} == {
        (split, angle) for split, angles in SPLITS.items() for angle in angles}
    near_far = list(csv.DictReader((output / "primary_near_far_ood_summary.csv").open()))
    assert {row["distance_group"] for row in near_far} == {"near", "far"}
    assert {row["ood_side"] for row in near_far} == {"ood_left", "ood_right"}
    across = list(csv.DictReader((output / "primary_summary_across_seeds.csv").open()))
    assert all(row["independent_seed_count"] == "5" for row in across)
    paired = list(csv.DictReader((output / "primary_paired_return_gaps.csv").open()))
    assert all(row["replicate_unit"] == "paired_end_to_end_seed" for row in paired)
    assert all(row["seed_count"] == "5" for row in paired)


def test_context_metric_episode_count_is_validated(tmp_path):
    _, root = _downstream_fixture(tmp_path)
    path = root / "results/pointrobot_primary/downstream/oracle/seed_0/context_metrics.csv"
    rows = list(csv.DictReader(path.open())); rows[0]["evaluation_episode_count"] = "9"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with pytest.raises(ValueError, match="episode count"): analyze(root / "results/pointrobot_primary", SPEC_PATH)


def test_deterministic_paired_bootstrap_and_exact_unique_seed_count():
    first = paired_bootstrap_interval([1, 2, 3, 4, 5], 20260806)
    second = paired_bootstrap_interval([1, 2, 3, 4, 5], 20260806)
    assert first == second
    with pytest.raises(ValueError, match="exactly five"): paired_bootstrap_interval([1, 2, 3, 4], 20260806)


def test_result_only_dependency_isolation():
    source = (ROOT / "src/crl_ood/analysis/analyze_pointrobot_primary.py").read_text()
    tree = ast.parse(source)
    imports = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imports |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert imports.isdisjoint({"gym", "gymnasium", "carl", "stable_baselines3", "torch"})


def test_every_protected_artifact_is_immutable_and_manifests_match():
    before = ROOT / "results/pointrobot_primary/protected_before.sha256"
    after = ROOT / "results/pointrobot_primary/protected_after.sha256"
    assert before.read_bytes() == after.read_bytes()
    paths = []
    for line in before.read_text().splitlines():
        expected, relative = line.split(None, 1); path = ROOT / relative.strip(); paths.append(path)
        assert path.is_file() and _sha(path) == expected
    assert any("configs/" in str(path.relative_to(ROOT)) for path in paths)
    for namespace in ("pointrobot_gate", "pointrobot_probe_audit", "pointrobot_encoders", "pointrobot_encoder_pilot_v1"):
        assert any(f"results/{namespace}/" in str(path.relative_to(ROOT)) for path in paths)
