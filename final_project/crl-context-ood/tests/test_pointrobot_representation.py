from __future__ import annotations

import ast
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from crl_ood.analysis.analyze_pointrobot_representation import analyze
from crl_ood.pointrobot_representation.evaluation import (
    circular_absolute_error, fit_linear_train_only, fit_pca_train_only, predict_linear,
)
from crl_ood.pointrobot_representation.manifest import verify_manifest, write_manifest
from crl_ood.pointrobot_representation.spec import (
    DATASET_SHA256, METHODS, SEEDS, SPLITS, build_checkpoint_jobs, dry_run_lines, load_spec,
    validate_primary_provenance,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "configs/pointrobot_representation/spec.yaml"


def _csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def _synthetic_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    spec_path = tmp_path / "spec.yaml"; spec_path.write_bytes(SPEC.read_bytes()); spec = load_spec(spec_path)
    results = tmp_path / "results/pointrobot_representation"; files = []
    latent_columns = {f"z_{index}": 0.0 for index in range(8)}
    for method_index, method in enumerate(METHODS):
        for seed in SEEDS:
            run = results / "evaluations" / method / f"seed_{seed}"; run.mkdir(parents=True)
            provenance = {"method": method, "encoder_seed": seed,
                "checkpoint_path": f"/frozen/{method}/seed_{seed}/best.pt", "checkpoint_sha256": str(seed) * 64,
                "dataset_checksum": DATASET_SHA256, "primary_execution_source_commit": "4" * 40,
                "authoritative_primary_source_snapshot": "6" * 40, "configuration_checksum": spec["_configuration_checksum"],
                "selection_role": "diagnostic_only", "checkpoint_reselected": False}
            provenance_path = run / "provenance.json"
            provenance_path.write_text(json.dumps(provenance), encoding="utf-8"); files.append(provenance_path)
            latent, predictions, state_rows, coordinates = [], [], [], []
            angle_rows, seed_rows = [], []
            trajectory_id = 0
            for split, angles in SPLITS.items():
                split_errors = []
                for angle in angles:
                    angle_errors = []
                    for timestep in (0, 1):
                        error = .04 + .01 * seed + .02 * method_index + .03 * abs(angle)
                        identity = {"method": method, "encoder_seed": seed, "split": split, "goal_angle": angle,
                            "trajectory_id": trajectory_id, "trajectory_seed": 41000, "timestep": timestep,
                            "history_length": min(timestep, 5)}
                        latent.append({**identity, "dataset_checksum": DATASET_SHA256,
                            "checkpoint_path": provenance["checkpoint_path"], "checkpoint_sha256": provenance["checkpoint_sha256"],
                            "source_commit": provenance["authoritative_primary_source_snapshot"],
                            "configuration_checksum": spec["_configuration_checksum"],
                            **{key: angle * (index + 1) + seed * .01 for index, key in enumerate(latent_columns)}})
                        predictions.append({**identity, "predicted_goal_angle": angle + error,
                            "circular_absolute_angle_error": error, "probe_fit_split": "train"})
                        state_rows.append({**identity, "current_state_x": timestep * .1, "current_state_y": timestep * .05,
                            "predicted_goal_angle": 0.0, "circular_absolute_angle_error": abs(angle),
                            "probe_fit_split": "train", "features": "current_state_only", "contains_history": False})
                        coordinates.append({**identity, "pc_1": angle + seed * .1, "pc_2": method_index + timestep * .1,
                            "pca_fit_split": "train", "standardization": "none", "cross_seed_alignment": False})
                        angle_errors.append(error); split_errors.append(error); trajectory_id += 1
                    angle_rows.append({"split": split, "goal_angle": angle,
                        "ood_distance_group": "near" if np.isclose(abs(angle), .8) else (
                            "far" if np.isclose(abs(angle), 1.0) else "not_ood"),
                        "circular_angle_mae": float(np.mean(angle_errors)), "sample_count": len(angle_errors),
                        "fit_split": "train", "method": method, "encoder_seed": seed})
                seed_rows.append({"split": split, "circular_angle_mae": float(np.mean(split_errors)),
                    "sample_count": len(split_errors), "fit_split": "train", "prediction_field": "predicted_goal_angle",
                    "method": method, "encoder_seed": seed})
            for name, rows in (("latent_samples.csv", latent), ("probe_predictions.csv", predictions),
                    ("probe_by_angle.csv", angle_rows), ("probe_by_seed.csv", seed_rows),
                    ("state_only_probe.csv", state_rows), ("pca_coordinates.csv", coordinates)):
                path = run / name; _csv(path, rows); files.append(path)
            pca_path = run / "pca_model.npz"
            np.savez(pca_path, mean=np.zeros(8), components=np.eye(8)[:2], explained_variance=np.array([2., 1.]),
                explained_variance_ratio=np.array([2 / 3, 1 / 3]), fit_split=np.asarray("train"),
                standardization=np.asarray("none")); files.append(pca_path)
            probe_path = run / "probe_model.npz"
            np.savez(probe_path, coefficients=np.zeros(9), state_only_coefficients=np.zeros(3), fit_split=np.asarray("train"))
            files.append(probe_path)
    write_manifest(results, files, results / "representation_evaluation_files.sha256")
    primary = tmp_path / "results/pointrobot_primary"
    primary_rows, closure_rows = [], []
    for method in ("no_context", "oracle", *METHODS):
        for seed in SEEDS:
            for split in SPLITS:
                base = -10 + seed * .1
                value = base + {"no_context": 0, "oracle": 8, "vae": 5, "contrastive": 6}[method]
                primary_rows.append({"method": method, "policy_seed": seed, "split": split, "mean_return": value})
                if method in METHODS and split == "id":
                    closure_rows.append({"method": method, "policy_seed": seed, "split": split,
                        "oracle_gap_closure": (value - base) / 8})
    _csv(primary / "primary_summary_by_seed.csv", primary_rows)
    _csv(primary / "primary_oracle_gap_closure.csv", closure_rows)
    return spec_path, results, primary


def test_exactly_ten_fixed_checkpoint_jobs_and_no_reselection():
    spec = load_spec(SPEC); jobs = build_checkpoint_jobs(spec, ROOT)
    assert len(jobs) == 10 and len({job.checkpoint for job in jobs}) == 10
    assert {(job.method, job.encoder_seed) for job in jobs} == {(method, seed) for method in METHODS for seed in SEEDS}
    assert spec["experiment"]["checkpoint_reselection"] == "forbidden"


def test_actual_checkpoint_hash_dataset_and_primary_provenance_validation():
    rows = validate_primary_provenance(load_spec(SPEC), ROOT)
    assert len(rows) == 10 and {row["dataset_checksum"] for row in rows} == {DATASET_SHA256}
    assert all(row["checkpoint_reselected"] is False for row in rows)


def test_exact_context_history_latent_and_diagnostic_protocol():
    spec = load_spec(SPEC)
    assert {name: tuple(values) for name, values in spec["contexts"].items()} == SPLITS
    assert spec["history"]["length"] == 5 and spec["history"]["current_state_outside_latent"]
    assert spec["history"]["empty_history_latent"] == "exact_zero_vector"
    assert spec["latent"]["dimension"] == 8
    assert spec["diagnostic_trajectories"]["action_sequences_shared_across_methods_seeds_and_contexts"]
    assert spec["diagnostic_trajectories"]["ppo_trajectories_forbidden"]


def test_train_only_linear_probe_and_circular_error():
    x = np.arange(16, dtype=float).reshape(8, 2); y = np.linspace(-.6, .6, 8)
    splits = np.array(["train"] * 5 + ["id", "ood_left", "ood_right"])
    coefficients = fit_linear_train_only(x, y, splits)
    assert coefficients.shape == (3,) and predict_linear(coefficients, x).shape == (8,)
    errors = circular_absolute_error(np.array([np.pi + .1, -np.pi - .2]), np.array([-np.pi + .1, np.pi - .2]))
    assert np.allclose(errors, 0.0, atol=1e-12)


def test_train_only_pca_is_independent_per_checkpoint():
    x = np.arange(64, dtype=float).reshape(8, 8); splits = np.array(["train"] * 5 + ["id"] * 3)
    model = fit_pca_train_only(x, splits)
    assert model["components"].shape == (2, 8)
    spec = load_spec(SPEC)
    assert spec["pca"]["fit_split"] == "train" and spec["pca"]["cross_seed_alignment"] == "forbidden"
    assert spec["pca"]["main_visualization_seed"] == 0


def test_dry_run_does_not_import_heavy_dependencies():
    code = ("import sys; from crl_ood.pointrobot_representation.spec import load_spec,dry_run_lines; "
        f"s=load_spec(r'{SPEC}'); print(len(dry_run_lines(s,r'{ROOT}'))); "
        "assert not any(x in sys.modules for x in ('torch','gym','gymnasium','carl','stable_baselines3'))")
    environment = dict(os.environ); environment["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_result_only_analysis_outputs_directional_ood_near_far_and_five_seeds(tmp_path):
    spec, results, primary = _synthetic_fixture(tmp_path)
    analyze(results, spec, primary)
    for name in (*CSV_OUTPUT_NAMES, "representation_findings.json", "representation_findings.md",
                 "representation_analysis_files.sha256", "representation_final_files.sha256"):
        assert (results / name).is_file()
    by_angle = _read(results / "representation_probe_by_angle.csv")
    assert {row["split"] for row in by_angle if row["ood_distance_group"] == "near"} == {"ood_left", "ood_right"}
    assert {row["split"] for row in by_angle if row["ood_distance_group"] == "far"} == {"ood_left", "ood_right"}
    summary = _read(results / "representation_probe_summary.csv")
    assert all(row["encoder_seed_count"] == "5" and row["context_count_is_replicate"] == "False" for row in summary)
    control = _read(results / "representation_control_by_seed.csv")
    assert {row["split"] for row in control} == {"id", "ood_left", "ood_right"}
    verify_manifest(results, results / "representation_final_files.sha256")


CSV_OUTPUT_NAMES = (
    "representation_latent_index.csv", "representation_probe_predictions.csv", "representation_probe_by_angle.csv",
    "representation_probe_by_seed.csv", "representation_probe_summary.csv", "representation_state_only_probe.csv",
    "representation_pca_coordinates.csv", "representation_pca_summary.csv",
    "representation_checkpoint_manifest.csv", "representation_control_by_seed.csv",
)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_state_only_baseline_has_no_history_and_pca_has_no_alignment(tmp_path):
    spec, results, primary = _synthetic_fixture(tmp_path); analyze(results, spec, primary)
    state = _read(results / "representation_state_only_probe.csv")
    assert all(row["features"] == "current_state_only" and row["contains_history"] == "False" for row in state)
    pca = _read(results / "representation_pca_summary.csv")
    assert all(row["fit_split"] == "train" and row["cross_seed_alignment"] == "False" for row in pca)


def test_result_only_dependency_import_and_subprocess_isolation(tmp_path):
    source = (ROOT / "src/crl_ood/analysis/analyze_pointrobot_representation.py").read_text()
    tree = ast.parse(source)
    imports = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imports |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert imports.isdisjoint({"torch", "gym", "gymnasium", "carl", "stable_baselines3"})
    code = ("import sys; import crl_ood.analysis.analyze_pointrobot_representation; "
        "assert not any(x in sys.modules for x in ('torch','gym','gymnasium','carl','stable_baselines3'))")
    environment = dict(os.environ); environment["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_evaluation_manifest_rejects_tampering(tmp_path):
    spec, results, _ = _synthetic_fixture(tmp_path)
    path = results / "evaluations/vae/seed_0/probe_by_seed.csv"
    path.write_text(path.read_text() + "tamper\n")
    with pytest.raises(ValueError, match="hash verification"):
        verify_manifest(results, results / "representation_evaluation_files.sha256")


def test_protected_artifacts_byte_identical():
    before = ROOT / "results/pointrobot_representation/protected_before.sha256"
    after = ROOT / "results/pointrobot_representation/protected_after.sha256"
    assert before.is_file() and after.is_file() and before.read_bytes() == after.read_bytes()
    for line in before.read_text().splitlines():
        expected, relative = line.split(None, 1)
        path = ROOT / relative.strip()
        import hashlib
        assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected
