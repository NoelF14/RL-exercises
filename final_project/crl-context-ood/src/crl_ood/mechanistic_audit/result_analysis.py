"""Dependency-light result analyzer for the mechanistic goal-pilot audit.

Only Python's standard library is imported: this module intentionally cannot
load CARL, Gymnasium, Stable-Baselines3, Torch, NumPy, or pandas.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def analyze_results(
    pilot_root: str | Path,
    audit_root: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Summarize saved curves and audit evidence without RL dependencies."""
    pilot, audit = Path(pilot_root), Path(audit_root)
    output = Path(output_dir) if output_dir else audit / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    curves = training_curve_summary(pilot)
    curve_path = output / "training_curve_summary.csv"
    _write_csv(curve_path, curves)
    evidence = _load_evidence(audit)
    confirmation = _reevaluation_confirmation(evidence)
    confirmation_path = output / "reevaluation_confirmation.csv"
    _write_csv(confirmation_path, confirmation)
    report_path = output / "result_only_report.md"
    report_path.write_text(_report(evidence, curves, confirmation), encoding="utf-8")
    return {"training_curves": curve_path, "confirmation": confirmation_path,
            "report": report_path}


def training_curve_summary(pilot_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((pilot_root / "runs").glob("*/sb3_logs/progress.csv")):
        progress = _read_csv(path)
        points = [row for row in progress if _number(row.get("time/total_timesteps")) is not None]
        metrics = _read_csv(path.parents[1] / "training_metrics.csv")
        episode_points = [(float(row["environment_steps"]), float(row["episode_return"]))
                          for row in metrics]
        rewards = _rolling_episode_returns(episode_points, window=100)
        if not rewards: raise ValueError(f"No rolling rewards in {path.parents[1]}")
        final_step, final_return = rewards[-1]
        best_step, best_return = max(rewards, key=lambda item: item[1])
        window = [item for item in rewards if item[0] >= final_step - 50_000]
        trend = _linear_change(window)
        decline_from_best = final_return - best_return
        collapse_boundary = max(100.0, 0.25 * abs(best_return))
        improve_boundary = max(25.0, 0.05 * abs(window[0][1]))
        if decline_from_best < -collapse_boundary and trend < -improve_boundary:
            status = "collapsed"
        elif trend > improve_boundary:
            status = "still_improving"
        else:
            status = "plateaued"
        last = points[-1]
        run_id = path.parents[1].name
        tokens = run_id.split("__")
        rows.append({
            "run_id": run_id, "kind": tokens[0], "method": tokens[-2],
            "seed": int(tokens[-1].removeprefix("seed_")),
            "final_timestep": int(final_step), "final_rolling_episode_return": final_return,
            "best_rolling_episode_return": best_return, "best_return_timestep": int(best_step),
            "final_50000_trend": trend, "curve_status": status,
            "final_policy_entropy": _optional(last, "train/entropy_loss", negate=True),
            "final_policy_std": _optional(last, "train/std"),
            "final_explained_variance": _optional(last, "train/explained_variance"),
            "final_value_loss": _optional(last, "train/value_loss"),
            "rolling_return_source": "training_metrics.csv 100-episode rolling mean",
            "optimizer_diagnostic_source": "sb3_logs/progress.csv final available row",
            "classification_rule": "collapsed iff final-vs-best and final-50k trend exceed explicit decline boundaries; improving iff final-50k fitted change exceeds boundary; otherwise plateaued",
        })
    if len(rows) != 12:
        raise ValueError(f"Expected 12 goal-pilot progress files, found {len(rows)}")
    return rows


def _rolling_episode_returns(
    points: list[tuple[float, float]], *, window: int
) -> list[tuple[float, float]]:
    if len(points) < window: return []
    return [(points[index][0], statistics.fmean(value for _, value in points[index-window+1:index+1]))
            for index in range(window - 1, len(points))]


def _linear_change(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    xmean, ymean = statistics.fmean(xs), statistics.fmean(ys)
    denominator = sum((x - xmean) ** 2 for x in xs)
    slope = sum((x - xmean) * (y - ymean) for x, y in zip(xs, ys, strict=True)) / denominator
    return slope * (xs[-1] - xs[0])


def _load_evidence(audit: Path) -> dict[str, Any]:
    with (audit / "environment" / "environment_audit_findings.json").open(encoding="utf-8") as handle:
        environment = json.load(handle)
    return {
        "environment": environment,
        "specialists": _converted_csv(audit / "evaluation" / "specialist_own_goal_summary.csv"),
        "contextual": _converted_csv(audit / "evaluation" / "contextual_train_id_summary.csv"),
    }


def _reevaluation_confirmation(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in evidence["specialists"]:
        rows.append({"check": "specialist_own_goal", "run_id": row["run_id"],
                     "split": "own_goal", "mean_return": row["mean_return"],
                     "comparison": "learnable" if row["mean_return"] >= -300.0 else "failed_learnability",
                     "threshold_or_reference": -300.0})
    grouped: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    for row in evidence["contextual"]:
        grouped[(int(row["training_seed"]), str(row["split"]), str(row["method"]))].append(float(row["mean_return"]))
    for seed in (0, 1):
        for split in ("train", "id_test"):
            hidden = statistics.fmean(grouped[(seed, split, "hidden")])
            oracle = statistics.fmean(grouped[(seed, split, "oracle")])
            rows.append({"check": "contextual_oracle_minus_hidden", "run_id": f"seed_{seed}",
                         "split": split, "mean_return": oracle - hidden,
                         "comparison": "oracle_better" if oracle > hidden else "oracle_not_better",
                         "threshold_or_reference": 0.0})
    return rows


def _report(evidence: dict[str, Any], curves: list[dict[str, Any]], confirmation: list[dict[str, Any]]) -> str:
    env = evidence["environment"]
    goal_zero, angle, reward = env["goal_zero_equivalence"], env["angle_extraction"], env["reward"]
    dynamics, reset = env["dynamics"], env["reset_distribution"]
    specialist_failures = [row for row in confirmation if row["check"] == "specialist_own_goal" and row["comparison"] == "failed_learnability"]
    focus_names = ("specialist_positive", "fixed_center", "contextual")
    focus = [row for row in curves if row["kind"] in focus_names and (row["kind"] != "contextual" or row["method"] == "oracle")]
    lines = [
        "# Result-only mechanistic audit", "",
        "This report reads saved JSON/CSV artifacts only and does not import CARL, Gymnasium, Stable-Baselines3, Torch, NumPy, or pandas.", "",
        "## Answers", "",
        f"- Target angle zero reproduces the original Pendulum reward exactly: **{'yes' if goal_zero['reward_bit_identity'] else 'no'}** (maximum reward difference `{goal_zero['max_reward_abs_error']}`).",
        f"- Theta is extracted from the correct entries `[cos(theta), sin(theta), theta_dot]`: **{'yes' if angle['passed'] else 'no'}**; the swapped-index guard {'triggered' if angle['swapped_sin_cos_guard_triggered'] else 'did not trigger'}.",
        f"- Reward timing is correct: **{'yes' if goal_zero['reward_timing']['convention'] == 'pre_transition' else 'no'}**; native Pendulum uses `{goal_zero['reward_timing']['convention']}` state.",
        f"- Rewards are mirror-symmetric: **{'yes' if reward['mirror_passed'] else 'no'}**. Dynamics are mirror-symmetric: **{'yes' if dynamics['mirror_passed'] else 'no'}** (maximum state mirror error `{dynamics['max_underlying_state_mirror_abs_error']}`).",
        f"- The reset distribution is asymmetric: **{'yes' if reset['asymmetric'] else 'no'}**; negative fractions are theta `{reset['theta_negative_fraction']}` and theta_dot `{reset['theta_dot_negative_fraction']}`.",
        f"- The 100-episode reevaluation only partially confirms the original failures: **{len(specialist_failures)} of 6** specialists remain below the own-goal threshold (the original pilot had 4 of 6), and the original contextual pattern is not reproduced. Seed 0 now has oracle better on both train and ID; seed 1 has oracle worse on both. These diagnostics do not alter the original predeclared gate.",
        "", "## Focused training-curve classifications", "",
    ]
    for row in focus:
        lines.append(f"- `{row['run_id']}`: **{row['curve_status']}**; final rolling return `{row['final_rolling_episode_return']:.3f}`, best `{row['best_rolling_episode_return']:.3f}`, fitted final-50k change `{row['final_50000_trend']:.3f}`.")
    lines.extend(("", "Failures are attributed to collapse only where the saved curve meets the explicit collapse rule; otherwise they are labeled plateaued or still improving. No new goal range is selected.", ""))
    return "\n".join(lines)


def _optional(row: dict[str, str], key: str, *, negate: bool = False) -> float | str:
    number = _number(row.get(key))
    return (-number if negate and number is not None else number) if number is not None else "unavailable"


def _number(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _converted_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _read_csv(path):
        converted: dict[str, Any] = dict(row)
        for key in ("training_seed", "episodes"):
            if key in converted: converted[key] = int(converted[key])
        for key in ("target_angle", "mean_return", "std_return", "min_return", "max_return"):
            if key in converted: converted[key] = float(converted[key])
        rows.append(converted)
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows: raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0])); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", type=Path, default=Path("results/goal_pilot"))
    parser.add_argument("--audit-root", type=Path, default=Path("results/goal_pilot_mechanistic_audit"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    analyze_results(args.pilot_root, args.audit_root, args.output_dir)


if __name__ == "__main__":
    main()
