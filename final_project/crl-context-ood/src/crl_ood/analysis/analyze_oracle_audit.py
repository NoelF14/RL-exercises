"""Analyze persisted oracle-audit artifacts without importing RL dependencies."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def analyze_oracle_audit(
    audit_root: str | Path,
    diagnostic_root: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Write descriptive answers and paired, seed-level audit tables."""
    audit_root = Path(audit_root)
    diagnostic_root = Path(diagnostic_root)
    output_dir = Path(output_dir) if output_dir else audit_root / "analysis"
    specialist = _read_csv(
        audit_root / "specialist_transfer/specialist_transfer_matrix_by_seed.csv"
    )
    ablation = _read_csv(audit_root / "oracle_ablation/oracle_ablation_by_seed.csv")
    _validate_specialist(specialist)
    _validate_ablation(ablation)
    output_dir.mkdir(parents=True, exist_ok=True)

    transfer_rows, transfer_answer = _specialist_transfer_answer(specialist)
    own_rows, own_answer = _specialist_own_context_answer(specialist)
    paired_rows, oracle_answer, sensitivity_answer = _oracle_answers(ablation)
    fixed_rows, fixed_answer = _fixed_center_answer(audit_root, diagnostic_root)
    findings = {
        "scope": {
            "inference": "descriptive_only_two_seeds",
            "confidence_intervals_computed": False,
            "ood_used_for_selection_or_tuning": False,
        },
        "questions": {
            "specialists_transfer_similarly": transfer_answer,
            "specialists_best_near_training_context": own_answer,
            "true_context_better_than_ablations": oracle_answer,
            "oracle_sensitive_to_scalar": sensitivity_answer,
            "fixed_center_oracle_matches_hidden": fixed_answer,
        },
    }
    paths = {
        "findings": output_dir / "oracle_audit_findings.json",
        "report": output_dir / "oracle_audit_findings.md",
        "specialist_spread": output_dir / "specialist_transfer_spread_by_context.csv",
        "specialist_best": output_dir / "specialist_best_context_by_seed.csv",
        "oracle_paired": output_dir / "oracle_ablation_paired_deltas.csv",
        "fixed_center": output_dir / "fixed_center_oracle_vs_hidden_by_seed.csv",
    }
    paths["findings"].write_text(json.dumps(findings, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(paths["specialist_spread"], transfer_rows)
    _write_csv(paths["specialist_best"], own_rows)
    _write_csv(paths["oracle_paired"], paired_rows)
    _write_csv(paths["fixed_center"], fixed_rows, fields=(
        "seed", "hidden_mean_return", "fixed_center_oracle_mean_return", "oracle_minus_hidden"
    ))
    paths["report"].write_text(_markdown(findings), encoding="utf-8")
    return paths


def _specialist_transfer_answer(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["seed"], row["evaluation_split"], row["evaluation_context_value"])].append(float(row["mean_return"]))
    spreads = [
        {
            "seed": seed, "evaluation_split": split, "evaluation_context_value": value,
            "best_minus_worst_specialist_return": max(values) - min(values),
        }
        for (seed, split, value), values in sorted(grouped.items())
    ]
    values = [row["best_minus_worst_specialist_return"] for row in spreads]
    answer = {
        "answer": "Report the observed spread; no similarity threshold was prespecified.",
        "mean_best_minus_worst_return": statistics.fmean(values),
        "max_best_minus_worst_return": max(values),
        "contexts_compared": len(spreads),
        "includes_ood_descriptive_only": True,
    }
    return spreads, answer


def _specialist_own_context_answer(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    targets = {"low": 0.8, "center": 1.0, "high": 1.2}
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["training_specialist_context"], row["seed"])].append(row)
    best_rows = []
    near_count = 0
    for (specialist, seed), candidates in sorted(grouped.items()):
        primary = [row for row in candidates if row["evaluation_split"] in {"train", "id_test"}]
        best = max(primary, key=lambda row: float(row["mean_return"]))
        values = sorted({float(row["evaluation_context_value"]) for row in primary})
        nearest_distance = min(abs(value - targets[specialist]) for value in values)
        distance = abs(float(best["evaluation_context_value"]) - targets[specialist])
        is_nearest = abs(distance - nearest_distance) < 1e-12
        near_count += int(is_nearest)
        best_rows.append(
            {
                "training_specialist_context": specialist,
                "seed": seed,
                "training_context_value": targets[specialist],
                "best_evaluation_split": best["evaluation_split"],
                "best_evaluation_context_value": best["evaluation_context_value"],
                "best_mean_return": best["mean_return"],
                "distance_from_training_context": distance,
                "best_is_nearest_available_context": is_nearest,
                "eligible_splits": "train,id_test",
            }
        )
    answer = {
        "answer": f"{near_count} of {len(best_rows)} specialist-seed pairs had their best mean return at the nearest available training context.",
        "nearest_count": near_count,
        "pairs": len(best_rows),
        "ood_not_used_for_best_context_selection": True,
    }
    return best_rows, answer


def _oracle_answers(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    indexed = {
        (row["seed"], row["evaluation_split"], row["observation_mode"]): float(row["mean_return"])
        for row in rows
    }
    paired = []
    for seed in ("0", "1"):
        for split in ("train", "id_test", "ood_low", "ood_high"):
            true = indexed[(seed, split, "true_context")]
            zero = indexed[(seed, split, "zero_context")]
            shuffled = indexed[(seed, split, "shuffled_context")]
            paired.append(
                {
                    "seed": seed, "evaluation_split": split,
                    "true_context_mean_return": true,
                    "zero_context_mean_return": zero,
                    "shuffled_context_mean_return": shuffled,
                    "true_minus_zero": true - zero,
                    "true_minus_shuffled": true - shuffled,
                    "zero_minus_shuffled": zero - shuffled,
                    "analysis_role": "descriptive_only" if split.startswith("ood") else "primary_audit",
                }
            )
    primary = [row for row in paired if row["analysis_role"] == "primary_audit"]
    better_both = sum(row["true_minus_zero"] > 0 and row["true_minus_shuffled"] > 0 for row in primary)
    deltas = [abs(row["true_minus_zero"]) for row in primary] + [abs(row["true_minus_shuffled"]) for row in primary]
    oracle = {
        "answer": f"True context beat both zero and shuffled context in {better_both} of {len(primary)} paired train/ID seed-split comparisons.",
        "paired_primary_comparisons": len(primary),
        "true_better_than_both_count": better_both,
        "ood_reported_descriptively_only": True,
    }
    sensitivity = {
        "answer": "The policy is observably scalar-sensitive." if any(delta > 1e-12 for delta in deltas) else "No return sensitivity to the scalar was observed.",
        "mean_absolute_primary_ablation_delta": statistics.fmean(deltas),
        "nonzero_tolerance": 1e-12,
    }
    return paired, oracle, sensitivity


def _fixed_center_answer(audit_root: Path, diagnostic_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for seed in (0, 1):
        oracle_path = audit_root / f"fixed_center_oracle_300k/length/oracle/seed_{seed}/context_returns.csv"
        hidden_path = diagnostic_root / f"default_300k/length/hidden/seed_{seed}/context_returns.csv"
        if not oracle_path.is_file():
            return [], {
                "answer": "Unavailable: both optional fixed-center oracle jobs have not produced validated result CSVs.",
                "available": False,
            }
        oracle = _single_mean(oracle_path)
        hidden = _single_mean(hidden_path)
        rows.append(
            {
                "seed": seed, "hidden_mean_return": hidden,
                "fixed_center_oracle_mean_return": oracle,
                "oracle_minus_hidden": oracle - hidden,
            }
        )
    deltas = [row["oracle_minus_hidden"] for row in rows]
    return rows, {
        "answer": "Performance difference is reported descriptively; no matching tolerance was prespecified.",
        "available": True,
        "mean_oracle_minus_hidden": statistics.fmean(deltas),
        "seed_deltas": deltas,
    }


def _validate_specialist(rows: list[dict[str, str]]) -> None:
    required = {"training_specialist_context", "seed", "evaluation_split", "evaluation_context_value", "mean_return"}
    if not rows or required - set(rows[0]):
        raise ValueError("Specialist transfer matrix is empty or incomplete")
    combinations = {(row["training_specialist_context"], row["seed"]) for row in rows}
    if combinations != {(label, str(seed)) for label in ("low", "center", "high") for seed in (0, 1)}:
        raise ValueError("Specialist transfer matrix lacks the six checkpoint seeds")


def _validate_ablation(rows: list[dict[str, str]]) -> None:
    expected = {(mode, str(seed), split) for mode in ("true_context", "zero_context", "shuffled_context") for seed in (0, 1) for split in ("train", "id_test", "ood_low", "ood_high")}
    actual = {(row["observation_mode"], row["seed"], row["evaluation_split"]) for row in rows}
    if actual != expected:
        raise ValueError("Oracle ablation results are not a complete paired matrix")


def _single_mean(path: Path) -> float:
    rows = _read_csv(path)
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one fixed-context row: {path}")
    return float(rows[0]["mean_return"])


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...] | None = None) -> None:
    if fields is None:
        if not rows:
            raise ValueError(f"Cannot infer CSV columns for empty output: {path}")
        fields = tuple(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _markdown(findings: dict[str, Any]) -> str:
    questions = findings["questions"]
    labels = (
        ("Do specialists transfer similarly across all lengths?", "specialists_transfer_similarly"),
        ("Does each specialist perform best near its training context?", "specialists_best_near_training_context"),
        ("Does true context outperform zero or shuffled context?", "true_context_better_than_ablations"),
        ("Is the oracle sensitive to its scalar?", "oracle_sensitive_to_scalar"),
        ("Does fixed-center oracle match the hidden baseline?", "fixed_center_oracle_matches_hidden"),
    )
    lines = ["# Oracle and context-necessity audit findings", "", "Two-seed descriptive analysis only. No confidence intervals; OOD is not used for tuning or selection.", ""]
    for index, (label, key) in enumerate(labels, start=1):
        lines.extend((f"{index}. {label}", "", str(questions[key]["answer"]), ""))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, default=Path("results/phase0_audit"))
    parser.add_argument("--diagnostic-root", type=Path, default=Path("results/phase0_diagnostic"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    for path in analyze_oracle_audit(args.audit_root, args.diagnostic_root, args.output_dir).values():
        print(path)


if __name__ == "__main__":
    main()
