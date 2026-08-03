"""Analyze goal-pilot CSV artifacts with no RL-framework imports."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

EXPECTED_KINDS = {
    "contextual": {"hidden", "oracle"},
    "specialist_negative": {"hidden"},
    "specialist_center": {"hidden"},
    "specialist_positive": {"hidden"},
    "fixed_center": {"oracle"},
}
TRAINING_GOAL = {
    "specialist_negative": -0.6,
    "specialist_center": 0.0,
    "specialist_positive": 0.6,
    "fixed_center": 0.0,
}
PRIMARY_SPLITS = ("train", "id_test")
OOD_SPLITS = ("ood_left", "ood_right")
OUTPUT_NAMES = (
    "goal_pilot_seed_results.csv",
    "goal_pilot_contextual_gaps_by_seed.csv",
    "goal_pilot_train_id_summary.csv",
    "goal_pilot_specialist_transfer_by_seed.csv",
    "goal_pilot_specialist_summary.csv",
    "goal_pilot_fixed_center_comparison.csv",
    "goal_pilot_ood_descriptive.csv",
    "goal_pilot_findings.md",
    "goal_pilot_contextual_return_by_goal.png",
    "goal_pilot_specialist_transfer_heatmap.png",
    "goal_pilot_paired_oracle_gap.png",
    "goal_pilot_training_curves.png",
)


def analyze_goal_pilot(
    results_root: str | Path,
    output_dir: str | Path | None = None,
    *,
    epsilon: float = 1e-8,
    specialist_learnable_return: float = -300.0,
    fixed_center_max_relative_degradation: float = 0.25,
    specialist_min_distinct_best_goals: int = 2,
) -> dict[str, Path]:
    """Validate all 12 results and write predeclared descriptive gate outputs."""
    root = Path(results_root)
    output = Path(output_dir) if output_dir else root / "analysis"
    seed_rows = _load_seed_results(root)
    _validate_complete_matrix(seed_rows)
    output.mkdir(parents=True, exist_ok=True)

    contextual = [row for row in seed_rows if row["kind"] == "contextual"]
    specialists = [row for row in seed_rows if row["kind"].startswith("specialist_")]
    contextual_gaps = _contextual_gaps(contextual, epsilon)
    train_id_summary = _train_id_summary(seed_rows)
    specialist_transfer = [row for row in specialists if row["split"] in PRIMARY_SPLITS]
    specialist_summary = _specialist_summary(specialist_transfer)
    fixed = _fixed_center(seed_rows, epsilon)
    ood = [dict(row, analysis_role="descriptive_only") for row in seed_rows if row["split"] in OOD_SPLITS]
    gates = _gates(
        specialists, contextual_gaps, fixed,
        specialist_learnable_return=specialist_learnable_return,
        fixed_center_max_relative_degradation=fixed_center_max_relative_degradation,
        specialist_min_distinct_best_goals=specialist_min_distinct_best_goals,
    )

    paths = {name: output / name for name in OUTPUT_NAMES}
    _write_csv(paths[OUTPUT_NAMES[0]], seed_rows)
    _write_csv(paths[OUTPUT_NAMES[1]], contextual_gaps)
    _write_csv(paths[OUTPUT_NAMES[2]], train_id_summary)
    _write_csv(paths[OUTPUT_NAMES[3]], specialist_transfer)
    _write_csv(paths[OUTPUT_NAMES[4]], specialist_summary)
    _write_csv(paths[OUTPUT_NAMES[5]], fixed)
    _write_csv(paths[OUTPUT_NAMES[6]], ood)
    paths[OUTPUT_NAMES[7]].write_text(_findings_markdown(gates), encoding="utf-8")
    _plots(root, contextual, specialist_summary, contextual_gaps, output)
    return paths


def _load_seed_results(root: Path) -> list[dict[str, Any]]:
    paths = sorted((root / "runs").glob("*/context_returns.csv"))
    if not paths:
        raise ValueError(f"No run context_returns.csv files found below {root / 'runs'}")
    rows: list[dict[str, Any]] = []
    for path in paths:
        for row in _read_csv(path):
            converted: dict[str, Any] = dict(row)
            for field in ("seed", "context_id", "episodes"):
                converted[field] = int(row[field])
            for field in ("target_angle", "normalized_target_angle", "mean_return", "std_return"):
                converted[field] = float(row[field])
            kind = str(converted["kind"])
            converted["training_target_angle"] = TRAINING_GOAL.get(kind, "multiple")
            converted["roles"] = (
                "specialist;fixed_center_hidden" if kind == "specialist_center"
                else "specialist" if kind.startswith("specialist_")
                else "fixed_center_oracle" if kind == "fixed_center"
                else "contextual"
            )
            rows.append(converted)
    return sorted(rows, key=lambda row: (str(row["kind"]), str(row["method"]), int(row["seed"]), str(row["split"]), float(row["target_angle"])))


def _validate_complete_matrix(rows: list[dict[str, Any]]) -> None:
    combinations = {(str(row["kind"]), str(row["method"]), int(row["seed"])) for row in rows}
    expected = {
        (kind, method, seed)
        for kind, methods in EXPECTED_KINDS.items()
        for method in methods for seed in (0, 1)
    }
    if combinations != expected:
        raise ValueError(f"Result matrix is incomplete or unexpected; expected {sorted(expected)}, got {sorted(combinations)}")
    for combination in expected:
        subset = [row for row in rows if (row["kind"], row["method"], row["seed"]) == combination]
        split_counts = {split: sum(row["split"] == split for row in subset) for split in (*PRIMARY_SPLITS, *OOD_SPLITS)}
        if split_counts != {"train": 5, "id_test": 4, "ood_left": 2, "ood_right": 2}:
            raise ValueError(f"Run {combination} does not contain the exact four goal splits: {split_counts}")


def _contextual_gaps(rows: list[dict[str, Any]], epsilon: float) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["seed"], row["split"], row["method"])].append(row["mean_return"])
    output = []
    for seed in (0, 1):
        for split in (*PRIMARY_SPLITS, *OOD_SPLITS):
            hidden = statistics.fmean(grouped[(seed, split, "hidden")])
            oracle = statistics.fmean(grouped[(seed, split, "oracle")])
            gap = oracle - hidden
            output.append(
                {"seed": seed, "split": split, "hidden_return": hidden, "oracle_return": oracle,
                 "oracle_minus_hidden": gap,
                 "relative_oracle_improvement": gap / (abs(hidden) + epsilon),
                 "analysis_role": "primary_gate" if split in PRIMARY_SPLITS else "descriptive_only"}
            )
    return output


def _train_id_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["split"] in PRIMARY_SPLITS:
            grouped[(row["kind"], row["method"], row["split"])].append(row["mean_return"])
    return [
        {"kind": kind, "method": method, "split": split, "observations": len(values),
         "mean_return": statistics.fmean(values), "std_return_descriptive": statistics.pstdev(values),
         "confidence_interval": "not_computed_two_seeds"}
        for (kind, method, split), values in sorted(grouped.items())
    ]


def _specialist_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, str, float], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(float(row["training_target_angle"]), row["split"], row["target_angle"])].append(row["mean_return"])
    return [
        {"training_target_angle": training, "evaluation_split": split,
         "evaluation_target_angle": target, "seed_count": len(values),
         "mean_return": statistics.fmean(values), "std_across_seeds_descriptive": statistics.pstdev(values)}
        for (training, split, target), values in sorted(grouped.items())
    ]


def _fixed_center(rows: list[dict[str, Any]], epsilon: float) -> list[dict[str, Any]]:
    output = []
    for seed in (0, 1):
        hidden = _one(rows, kind="specialist_center", method="hidden", seed=seed, split="train", target=0.0)
        oracle = _one(rows, kind="fixed_center", method="oracle", seed=seed, split="train", target=0.0)
        gap = oracle["mean_return"] - hidden["mean_return"]
        output.append(
            {"seed": seed, "target_angle": 0.0, "hidden_run_id": hidden["run_id"],
             "hidden_reuse": "specialist_center", "oracle_run_id": oracle["run_id"],
             "hidden_return": hidden["mean_return"], "oracle_return": oracle["mean_return"],
             "oracle_minus_hidden": gap, "relative_oracle_improvement": gap / (abs(hidden["mean_return"]) + epsilon)}
        )
    return output


def _one(rows: list[dict[str, Any]], *, kind: str, method: str, seed: int, split: str, target: float) -> dict[str, Any]:
    found = [row for row in rows if row["kind"] == kind and row["method"] == method and row["seed"] == seed and row["split"] == split and math.isclose(row["target_angle"], target, abs_tol=1e-12)]
    if len(found) != 1:
        raise ValueError(f"Expected one row for {kind}/{method}/seed={seed}/{split}/goal={target}")
    return found[0]


def _gates(
    specialists: list[dict[str, Any]], gaps: list[dict[str, Any]], fixed: list[dict[str, Any]],
    *, specialist_learnable_return: float, fixed_center_max_relative_degradation: float,
    specialist_min_distinct_best_goals: int,
) -> dict[str, Any]:
    own = []
    best = []
    for seed in (0, 1):
        seed_best = []
        for kind, target in (("specialist_negative", -0.6), ("specialist_center", 0.0), ("specialist_positive", 0.6)):
            row = _one(specialists, kind=kind, method="hidden", seed=seed, split="train", target=target)
            own.append({"seed": seed, "kind": kind, "return": row["mean_return"], "pass": row["mean_return"] >= specialist_learnable_return})
            candidates = [item for item in specialists if item["kind"] == kind and item["seed"] == seed and item["split"] in PRIMARY_SPLITS]
            winner = max(candidates, key=lambda item: item["mean_return"])
            seed_best.append(winner["target_angle"])
            best.append({"seed": seed, "kind": kind, "best_target_angle": winner["target_angle"], "best_return": winner["mean_return"]})
    distinct_by_seed = {seed: len({row["best_target_angle"] for row in best if row["seed"] == seed}) for seed in (0, 1)}
    contextual_primary = [row for row in gaps if row["split"] in PRIMARY_SPLITS]
    checks = {
        "specialists_learnable": all(row["pass"] for row in own),
        "specialists_target_dependent": all(value >= specialist_min_distinct_best_goals for value in distinct_by_seed.values()),
        "contextual_oracle_better_train_and_id_both_seeds": all(row["oracle_minus_hidden"] > 0 for row in contextual_primary),
        "fixed_center_no_large_oracle_degradation": all(row["relative_oracle_improvement"] >= -fixed_center_max_relative_degradation for row in fixed),
    }
    return {
        "checks": checks, "accepted": all(checks.values()), "own_goal_results": own,
        "specialist_best_goals": best, "distinct_best_goals_by_seed": distinct_by_seed,
        "contextual_primary": contextual_primary, "fixed_center": fixed,
        "thresholds": {"specialist_learnable_return": specialist_learnable_return,
                       "specialist_min_distinct_best_goals": specialist_min_distinct_best_goals,
                       "fixed_center_max_relative_degradation": fixed_center_max_relative_degradation},
    }


def _findings_markdown(gates: dict[str, Any]) -> str:
    checks = gates["checks"]
    lines = [
        "# Goal-pilot context-necessity findings", "",
        "Two-seed descriptive gate analysis only. No confidence interval is computed. OOD-left and OOD-right are reported separately and never affect acceptance.", "",
        f"Overall predeclared gate: **{'ACCEPT' if gates['accepted'] else 'REJECT'}**", "",
        "## Train/ID decision checks", "",
    ]
    labels = {
        "specialists_learnable": "All three fixed-context tasks meet the own-goal learnability threshold for both seeds",
        "specialists_target_dependent": "Specialists show target-dependent best-goal performance for both seeds",
        "contextual_oracle_better_train_and_id_both_seeds": "Contextual oracle beats hidden on train and ID for both seeds",
        "fixed_center_no_large_oracle_degradation": "Fixed-center oracle avoids the predeclared large degradation",
    }
    for key, label in labels.items():
        lines.append(f"- {'PASS' if checks[key] else 'FAIL'}: {label}.")
    lines.extend(("", "Relative oracle improvement is `(oracle_return - hidden_return) / (abs(hidden_return) + epsilon)`.", "", "## Preconditions and thresholds", ""))
    for key, value in gates["thresholds"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(("", "OOD-left and OOD-right are descriptive only and were excluded from every check.", ""))
    return "\n".join(lines)


def _plots(root: Path, contextual: list[dict[str, Any]], specialist_summary: list[dict[str, Any]],
           gaps: list[dict[str, Any]], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for method in ("hidden", "oracle"):
        grouped: dict[float, list[float]] = defaultdict(list)
        for row in contextual:
            grouped[row["target_angle"]].append(row["mean_return"])
        # Rebuild per-method to avoid mixing the two curves.
        grouped = defaultdict(list)
        for row in contextual:
            if row["method"] == method:
                grouped[row["target_angle"]].append(row["mean_return"])
        xs = sorted(grouped)
        ax.plot(xs, [statistics.fmean(grouped[x]) for x in xs], marker="o", label=method)
    ax.set(xlabel="Target angle (rad)", ylabel="Mean return", title="Contextual return by target goal")
    ax.legend(); fig.tight_layout(); fig.savefig(output / OUTPUT_NAMES[8], dpi=160); plt.close(fig)

    train_targets = (-0.6, 0.0, 0.6)
    eval_targets = sorted({row["evaluation_target_angle"] for row in specialist_summary if row["evaluation_split"] in PRIMARY_SPLITS})
    lookup = {(row["training_target_angle"], row["evaluation_target_angle"]): row["mean_return"] for row in specialist_summary}
    matrix = [[lookup[(train, target)] for target in eval_targets] for train in train_targets]
    fig, ax = plt.subplots(figsize=(8, 3.5)); image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set(xticks=range(len(eval_targets)), xticklabels=eval_targets, yticks=range(3), yticklabels=train_targets,
           xlabel="Evaluation target (rad)", ylabel="Training target (rad)", title="Specialist train/ID transfer")
    fig.colorbar(image, ax=ax, label="Mean return"); fig.tight_layout(); fig.savefig(output / OUTPUT_NAMES[9], dpi=160); plt.close(fig)

    primary = [row for row in gaps if row["split"] in PRIMARY_SPLITS]
    fig, ax = plt.subplots(figsize=(6, 4)); labels = [f"s{r['seed']} {r['split']}" for r in primary]
    ax.bar(labels, [row["oracle_minus_hidden"] for row in primary]); ax.axhline(0, color="black", linewidth=0.8)
    ax.set(ylabel="Oracle − hidden return", title="Paired contextual oracle gaps"); ax.tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(output / OUTPUT_NAMES[10], dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for path in sorted((root / "runs").glob("*/training_metrics.csv")):
        rows = _read_csv(path)
        if rows:
            ax.plot([float(row["environment_steps"]) for row in rows], [float(row["episode_return"]) for row in rows], alpha=0.45, label=path.parent.name)
    ax.set(xlabel="Environment steps", ylabel="Episode return", title="Goal-pilot training curves")
    if len(ax.lines) <= 12:
        ax.legend(fontsize=5, ncol=2)
    fig.tight_layout(); fig.savefig(output / OUTPUT_NAMES[11], dpi=160); plt.close(fig)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty required output {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results/goal_pilot"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    for path in analyze_goal_pilot(args.results_root, args.output_dir).values():
        print(path)


if __name__ == "__main__":
    main()
