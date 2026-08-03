from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

from crl_ood.analysis.analyze_goal_pilot import OUTPUT_NAMES, analyze_goal_pilot

SPLITS = {
    "train": [-0.6, -0.3, 0.0, 0.3, 0.6],
    "id_test": [-0.45, -0.15, 0.15, 0.45],
    "ood_left": [-1.0, -0.8],
    "ood_right": [0.8, 1.0],
}
RUNS = (
    ("contextual", "hidden", None),
    ("contextual", "oracle", None),
    ("specialist_negative", "hidden", -0.6),
    ("specialist_center", "hidden", 0.0),
    ("specialist_positive", "hidden", 0.6),
    ("fixed_center", "oracle", 0.0),
)


def test_result_only_analyzer_dependency_isolation():
    code = (
        "import sys; blocked={'carl','gym','gymnasium','stable_baselines3','torch'}; "
        "sys.meta_path.insert(0,type('B',(),{'find_spec':lambda s,n,p=None,t=None: "
        "(_ for _ in ()).throw(RuntimeError(n)) if n.split('.')[0] in blocked else None})()); "
        "import crl_ood.analysis.analyze_goal_pilot; print('isolated')"
    )
    completed = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    assert completed.stdout.strip() == "isolated"


def test_synthetic_pass_fixture_writes_all_tables_findings_and_plots(tmp_path):
    root = _synthetic_results(tmp_path / "goal_pilot")
    paths = analyze_goal_pilot(root)
    assert set(paths) == set(OUTPUT_NAMES)
    for path in paths.values():
        assert path.is_file() and path.stat().st_size > 0
    findings = paths["goal_pilot_findings.md"].read_text(encoding="utf-8")
    assert "Overall predeclared gate: **ACCEPT**" in findings
    assert "No confidence interval" in findings
    assert "OOD-left and OOD-right" in findings
    ood = _read(paths["goal_pilot_ood_descriptive.csv"])
    assert {row["split"] for row in ood} == {"ood_left", "ood_right"}
    assert {row["analysis_role"] for row in ood} == {"descriptive_only"}
    gaps = _read(paths["goal_pilot_contextual_gaps_by_seed.csv"])
    assert len(gaps) == 8
    assert all(float(row["relative_oracle_improvement"]) > 0 for row in gaps)
    fixed = _read(paths["goal_pilot_fixed_center_comparison.csv"])
    assert {row["hidden_reuse"] for row in fixed} == {"specialist_center"}
    assert all("specialist_center" in row["hidden_run_id"] for row in fixed)


def test_synthetic_gate_failure_uses_train_id_not_ood(tmp_path):
    root = _synthetic_results(tmp_path / "goal_pilot")
    contextual_oracle = next((root / "runs").glob("contextual__oracle__seed_0/context_returns.csv"))
    rows = _read(contextual_oracle)
    for row in rows:
        if row["split"] == "id_test":
            row["mean_return"] = "-500"
    _write(contextual_oracle, rows)
    paths = analyze_goal_pilot(root, root / "failed_analysis")
    findings = paths["goal_pilot_findings.md"].read_text(encoding="utf-8")
    assert "Overall predeclared gate: **REJECT**" in findings
    assert "FAIL: Contextual oracle beats hidden on train and ID for both seeds" in findings


def _synthetic_results(root: Path) -> Path:
    for kind, method, training_target in RUNS:
        for seed in (0, 1):
            run_id = f"{kind}__{method}__seed_{seed}"
            run_dir = root / "runs" / run_id
            run_dir.mkdir(parents=True)
            rows = []
            for split, goals in SPLITS.items():
                for context_id, goal in enumerate(goals):
                    if kind == "contextual":
                        value = -200.0 - 5.0 * abs(goal) + seed
                        if method == "oracle":
                            value += 20.0
                    elif kind.startswith("specialist_"):
                        value = -100.0 - 150.0 * abs(goal - float(training_target)) + seed
                    else:
                        value = -105.0 - 150.0 * abs(goal) + seed
                    rows.append(
                        {"run_id": run_id, "method": method, "seed": seed, "kind": kind,
                         "split": split, "context_id": context_id, "target_angle": goal,
                         "normalized_target_angle": goal / 0.6, "episodes": 5,
                         "mean_return": value, "std_return": 1.0}
                    )
            _write(run_dir / "context_returns.csv", rows)
            _write(
                run_dir / "training_metrics.csv",
                [{"run_id": run_id, "method": method, "seed": seed, "kind": kind,
                  "environment_steps": step, "episode_index": index,
                  "episode_return": -300 + index * 10, "episode_length": 200}
                 for index, step in enumerate((200, 400, 600))],
            )
    return root


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader(); writer.writerows(rows)
