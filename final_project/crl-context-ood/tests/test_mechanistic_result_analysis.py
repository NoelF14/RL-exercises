from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

from crl_ood.mechanistic_audit.result_analysis import analyze_results

ROOT = Path(__file__).parents[1]


def _write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0])); writer.writeheader(); writer.writerows(rows)


def test_result_analyzer_has_only_standard_library_imports():
    path = ROOT / "src/crl_ood/mechanistic_audit/result_analysis.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots = {alias.name.split(".")[0] for node in ast.walk(tree)
             if isinstance(node, ast.Import) for alias in node.names}
    roots |= {node.module.split(".")[0] for node in ast.walk(tree)
              if isinstance(node, ast.ImportFrom) and node.module}
    assert roots.isdisjoint({"carl", "gym", "gymnasium", "stable_baselines3", "torch", "numpy", "pandas"})


def test_synthetic_result_analysis_writes_curve_and_confirmation_outputs(tmp_path):
    pilot, audit = tmp_path / "pilot", tmp_path / "audit"
    run_specs = []
    for seed in (0, 1):
        run_specs.extend([
            (f"contextual__all_train__hidden__seed_{seed}", "hidden"),
            (f"contextual__all_train__oracle__seed_{seed}", "oracle"),
            (f"specialist_negative__goal_neg_0p6__hidden__seed_{seed}", "hidden"),
            (f"specialist_center__goal_zero_0p0__hidden__seed_{seed}", "hidden"),
            (f"specialist_positive__goal_pos_0p6__hidden__seed_{seed}", "hidden"),
            (f"fixed_center__goal_zero_0p0__oracle__seed_{seed}", "oracle"),
        ])
    for run_id, _ in run_specs:
        _write(pilot / "runs" / run_id / "sb3_logs/progress.csv", [
            {"time/total_timesteps": 250000, "rollout/ep_rew_mean": -500,
             "train/entropy_loss": -1.4, "train/std": 1.0,
             "train/explained_variance": 0.2, "train/value_loss": 3.0},
            {"time/total_timesteps": 300000, "rollout/ep_rew_mean": -400,
             "train/entropy_loss": -1.3, "train/std": .9,
             "train/explained_variance": .4, "train/value_loss": 2.0},
        ])
        _write(pilot / "runs" / run_id / "training_metrics.csv", [
            {"environment_steps": (index + 1) * 3000, "episode_return": -500 + index}
            for index in range(100)
        ])
    environment = {
        "goal_zero_equivalence": {"reward_bit_identity": True, "max_reward_abs_error": 0.0,
                                  "reward_timing": {"convention": "pre_transition"}},
        "angle_extraction": {"passed": True, "swapped_sin_cos_guard_triggered": True},
        "reward": {"mirror_passed": True},
        "dynamics": {"mirror_passed": True, "max_underlying_state_mirror_abs_error": 0.0},
        "reset_distribution": {"asymmetric": True, "theta_negative_fraction": 0.0,
                               "theta_dot_negative_fraction": 0.0},
    }
    (audit / "environment").mkdir(parents=True)
    (audit / "environment/environment_audit_findings.json").write_text(json.dumps(environment), encoding="utf-8")
    specialist = [{"run_id": f"s{i}", "kind": "specialist", "method": "hidden",
                   "training_seed": i % 2, "split": "own_goal", "target_angle": 0,
                   "episodes": 100, "mean_return": -400, "std_return": 1,
                   "min_return": -402, "max_return": -398} for i in range(6)]
    contextual = []
    for seed in (0, 1):
        for split in ("train", "id_test"):
            for method, value in (("hidden", -100), ("oracle", -120)):
                contextual.append({"run_id": f"c-{seed}-{method}", "kind": "contextual",
                                   "method": method, "training_seed": seed, "split": split,
                                   "target_angle": 0, "episodes": 100, "mean_return": value,
                                   "std_return": 1, "min_return": value-2, "max_return": value+2})
    _write(audit / "evaluation/specialist_own_goal_summary.csv", specialist)
    _write(audit / "evaluation/contextual_train_id_summary.csv", contextual)
    outputs = analyze_results(pilot, audit)
    assert all(path.is_file() for path in outputs.values())
    assert "does not import CARL" in outputs["report"].read_text(encoding="utf-8")
