"""Result-only PointRobot probe-audit v2 analyzer.

This module deliberately imports no Gym/Gymnasium, CARL, Stable-Baselines3, or Torch.
It reads only saved CSV, JSON, and YAML inputs.
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

import yaml

OUTPUTS = (
    "behavior_geometry_summary.csv", "probe_model_summary.csv",
    "probe_v1_vs_v2_comparison.csv", "pointrobot_probe_audit_findings.json",
    "pointrobot_probe_audit_findings.md", "analytic_error_vs_history.png",
    "full_rank_fraction_vs_history.png", "probe_model_error_vs_history.png",
    "condition_number_distribution.png",
)


def analyze(results_dir: str | Path, config_path: str | Path) -> dict[str, Path]:
    root, config_path = Path(results_dir), Path(config_path)
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    analytic = _read_csv(root / "analytic_estimator_by_history.csv")
    models = _read_csv(root / "probe_model_results_by_seed.csv")
    with (root / "probe_alignment_audit.json").open(encoding="utf-8") as handle:
        alignment = json.load(handle)
    source_root = Path(config["experiment"]["source_results"])
    with (source_root / "analysis" / "pointrobot_gate_findings.json").open(encoding="utf-8") as handle:
        v1 = json.load(handle)
    geometry = geometry_summary(analytic)
    model_summary = summarize_models(models)
    findings = evaluate_v2(geometry, model_summary, alignment, config, v1)
    comparison = compare_v1_v2(v1, findings, source_root)
    paths = {name: root / name for name in OUTPUTS}
    _write_csv(paths[OUTPUTS[0]], geometry)
    _write_csv(paths[OUTPUTS[1]], model_summary)
    _write_csv(paths[OUTPUTS[2]], comparison)
    paths[OUTPUTS[3]].write_text(json.dumps(findings, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths[OUTPUTS[4]].write_text(_markdown(findings), encoding="utf-8")
    _plots(analytic, geometry, model_summary, paths)
    return paths


def geometry_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["behavior_policy"], int(row["history_length"]), row["split"])].append(row)
    output = []
    for (policy, history, split), values in sorted(groups.items()):
        conditions = [float(row["condition_number"]) for row in values]
        finite = sorted(value for value in conditions if math.isfinite(value))
        output.append({
            "behavior_policy": policy, "history_length": history, "split": split,
            "history_count": len(values),
            "full_rank_fraction": statistics.fmean(_truth(row["geometrically_identifiable"]) for row in values),
            "condition_number_median": _quantile(finite, 0.50),
            "condition_number_p90": _quantile(finite, 0.90),
            "condition_number_p99": _quantile(finite, 0.99),
            "infinite_condition_fraction": statistics.fmean(not math.isfinite(value) for value in conditions),
            "mean_circular_angle_error": statistics.fmean(float(row["circular_angle_error"]) for row in values),
            "analysis_role": "audit_decision" if split == "id" else
                             "descriptive_only" if split.startswith("ood_") else "training_diagnostic",
        })
    return output


def summarize_models(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["model"], int(row["history_length"]), row["split"])].append(row)
    output = []
    for (model, history, split), values in sorted(groups.items()):
        errors = [float(row["circular_angle_mae"]) for row in values]
        output.append({
            "model": model, "history_length": history, "split": split,
            "seed_count": len(values), "mean_circular_angle_mae": statistics.fmean(errors),
            "std_across_seeds_descriptive": statistics.pstdev(errors),
            "parameter_count_min": min(int(row["parameter_count"]) for row in values),
            "parameter_count_max": max(int(row["parameter_count"]) for row in values),
            "analysis_role": "audit_decision" if split == "id" else
                             "descriptive_only" if split.startswith("ood_") else "training_diagnostic",
        })
    return output


def evaluate_v2(geometry: list[dict[str, Any]], models: list[dict[str, Any]],
                alignment: dict[str, Any], config: dict[str, Any], v1: dict[str, Any]) -> dict[str, Any]:
    audit = config["audit"]
    selected_policy = audit["selected_behavior_policy"]
    id_geometry = sorted((row for row in geometry if row["behavior_policy"] == selected_policy
                          and row["split"] == "id"), key=lambda row: int(row["history_length"]))
    eligible = [row for row in id_geometry if int(row["history_length"]) >= 2
                and float(row["full_rank_fraction"]) >= float(audit["analytic_id_min_full_rank_fraction"])]
    first = eligible[0] if eligible else None
    analytic_error_pass = bool(first and float(first["mean_circular_angle_error"])
                               <= float(audit["analytic_id_max_circular_mae"]))
    analytic_rank_pass = bool(first and float(first["full_rank_fraction"])
                              >= float(audit["analytic_id_min_full_rank_fraction"]))
    lookup = {(row["model"], int(row["history_length"])): float(row["mean_circular_angle_mae"])
              for row in models if row["split"] == "id"}
    state = lookup[("state_only", 0)]
    candidates = []
    for model in audit["nonlinear_candidates"]:
        h1 = lookup[(model, 1)]
        for history in map(int, audit["nonlinear_history_candidates"]):
            error = lookup[(model, history)]
            reduction = (state - error) / state if state else 0.0
            beyond = (h1 - error) / h1 if h1 else 0.0
            candidates.append({
                "model": model, "history_length": history, "id_circular_mae": error,
                "state_only_id_circular_mae": state, "h1_id_circular_mae": h1,
                "relative_reduction_vs_state": reduction,
                "relative_improvement_vs_h1": beyond,
                "pass": reduction >= float(audit["nonlinear_min_relative_reduction_vs_state"])
                        and beyond >= float(audit["nonlinear_min_relative_improvement_vs_h1"]),
            })
    nonlinear_pass = any(item["pass"] for item in candidates)
    alignment_pass = all(bool(alignment.get(key)) for key in (
        "reward_alignment_pass", "no_future_transitions_pass", "target_constant_within_episode_pass",
        "forbidden_input_fields_absent_pass", "matched_actions_across_contexts_pass",
        "train_id_ood_fit_isolation_pass", "dataset_parity_pass",
    ))
    criteria = {
        "analytic_id_circular_mae": analytic_error_pass,
        "analytic_id_full_rank_fraction": analytic_rank_pass,
        "nonlinear_reduction_vs_state_and_h1": nonlinear_pass,
        "leakage_and_alignment": alignment_pass,
    }
    return {
        "audit_name": "PointRobot probe-audit v2", "accepted": all(criteria.values()),
        "criteria": criteria, "thresholds": audit,
        "selected_behavior_policy_prespecified": selected_policy,
        "analytic_decision_history": first, "nonlinear_candidates": candidates,
        "passing_nonlinear_candidates": [item for item in candidates if item["pass"]],
        "alignment": alignment,
        "primary_decision_split": "id", "training_split": "train",
        "ood_excluded_from_acceptance": ["ood_left", "ood_right"],
        "original_pointrobot_gate_v1": {"accepted": bool(v1["accepted"]), "unchanged": True},
        "diagnostic_model_scope": "MLP and GRU are supervised probes only; neither is a VAE or contrastive encoder.",
    }


def compare_v1_v2(v1: dict[str, Any], v2: dict[str, Any], source_root: Path) -> list[dict[str, Any]]:
    old_probe = v1["supporting_values"]["probe"]
    rows = []
    for item in old_probe:
        values = [("state_only_id_mae", item["state_only_id_mae"]),
                  ("history_h1_id_mae", item["history_h1_id_mae"])]
        values.extend((f"history_h{history}_id_mae", value)
                      for history, value in sorted(item["long_history_id_mae"].items(), key=lambda pair: int(pair[0])))
        values.extend((("best_relative_reduction_vs_state", item["best_relative_reduction_vs_state"]),
                       ("relative_improvement_beyond_h1", item["relative_improvement_beyond_h1"])))
        for metric, value in values:
            rows.append({
                "decision_version": "v1", "scope": "original_predeclared_gate_raw_ridge",
                "seed_or_aggregate": item["seed"], "history_or_criterion": metric,
                "value": value,
                "pass": bool(item["long_history_pass"] and item["beyond_h1_pass"]),
                "overall_accepted": bool(v1["accepted"]), "source": str(source_root / "analysis" / "pointrobot_gate_findings.json"),
            })
    for criterion, passed in v2["criteria"].items():
        rows.append({
            "decision_version": "v2", "scope": "probe_validity_and_trajectory_identifiability",
            "seed_or_aggregate": "aggregate_train_id_only", "history_or_criterion": criterion,
            "value": "", "pass": passed, "overall_accepted": v2["accepted"],
            "source": "results/pointrobot_probe_audit/pointrobot_probe_audit_findings.json",
        })
    return rows


def _markdown(findings: dict[str, Any]) -> str:
    analytic = findings["analytic_decision_history"] or {}
    lines = [
        "# PointRobot probe-audit v2 findings", "",
        f"Probe-audit v2: **{'PASS' if findings['accepted'] else 'FAIL'}**.", "",
        "The original PointRobot gate v1 remains unchanged: **REJECT**. This v2 audit is a separate assessment of probe validity and trajectory identifiability.", "",
        "## Identifiability derivation", "",
        "With `r_t = -(||p_{t+1}-g||^2 + lambda||a_t||^2)` and `||g||=1`, expansion gives",
        "`p_{t+1}^T g = (r_t + ||p_{t+1}||^2 + 1 + lambda||a_t||^2)/2 = b_t`.",
        "Stacking history rows gives `P g = b`; the audit uses `pinv(P)b` and reports rank and conditioning.", "",
        "## Decision criteria", "",
    ]
    lines.extend(f"- {'PASS' if value else 'FAIL'}: `{key}`" for key, value in findings["criteria"].items())
    lines.extend([
        "", "## Decision values", "",
        f"Prespecified behavior policy: `{findings['selected_behavior_policy_prespecified']}`.",
        f"First eligible history: H={analytic.get('history_length', 'none')}; full-rank fraction={analytic.get('full_rank_fraction', 'n/a')}; ID circular MAE={analytic.get('mean_circular_angle_error', 'n/a')}.",
        "OOD-left and OOD-right are descriptive only and never enter acceptance.",
        "MLP and GRU results are supervised diagnostics, separate from any later VAE or contrastive encoder.",
    ])
    return "\n".join(lines) + "\n"


def _plots(analytic: list[dict[str, str]], geometry: list[dict[str, Any]], models: list[dict[str, Any]], paths: dict[str, Path]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    id_geometry = [row for row in geometry if row["split"] == "id"]
    for metric, output_name, ylabel in (
        ("mean_circular_angle_error", "analytic_error_vs_history.png", "ID circular-angle MAE (rad)"),
        ("full_rank_fraction", "full_rank_fraction_vs_history.png", "ID full-rank fraction"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4))
        for policy in sorted({row["behavior_policy"] for row in id_geometry}):
            values = sorted((row for row in id_geometry if row["behavior_policy"] == policy),
                            key=lambda row: int(row["history_length"]))
            ax.plot([row["history_length"] for row in values], [row[metric] for row in values], marker="o", label=policy)
        ax.set(xlabel="History length", ylabel=ylabel, title=ylabel + " by behavior policy")
        ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(paths[output_name], dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for model in ("raw_ridge", "engineered_linear", "mlp", "gru"):
        values = sorted((row for row in models if row["split"] == "id" and row["model"] == model),
                        key=lambda row: int(row["history_length"]))
        ax.plot([row["history_length"] for row in values], [row["mean_circular_angle_mae"] for row in values], marker="o", label=model)
    state = next(row for row in models if row["split"] == "id" and row["model"] == "state_only")
    ax.axhline(float(state["mean_circular_angle_mae"]), color="black", linestyle="--", label="state_only")
    ax.set(xlabel="History length", ylabel="ID circular-angle MAE (rad)", title="Probe model audit")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(paths["probe_model_error_vs_history.png"], dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels, distributions = [], []
    for policy in sorted({row["behavior_policy"] for row in id_geometry}):
        for history in (2, 5, 10):
            values = [float(row["condition_number"]) for row in analytic
                      if row["split"] == "id" and row["behavior_policy"] == policy
                      and int(row["history_length"]) == history and math.isfinite(float(row["condition_number"]))]
            labels.append(f"{policy}\nH{history}")
            distributions.append(values)
    ax.boxplot(distributions, tick_labels=labels, showfliers=False)
    ax.set_yscale("log")
    ax.tick_params(axis="x", labelrotation=35, labelsize=7)
    ax.set(ylabel="Condition number (log scale)", title="Position-design condition-number distributions (ID)")
    fig.tight_layout(); fig.savefig(paths["condition_number_distribution.png"], dpi=160); plt.close(fig)


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        return float("inf")
    position = fraction * (len(values) - 1)
    low, high = math.floor(position), math.ceil(position)
    return values[low] if low == high else values[low] + (position - low) * (values[high] - values[low])


def _truth(value: Any) -> bool:
    return str(value).lower() in {"1", "true", "yes"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() != ".csv":
        raise ValueError("Result-only analyzer accepts CSV inputs only here")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/pointrobot_probe_audit"))
    parser.add_argument("--config", type=Path, default=Path("configs/pointrobot_probe_audit/audit.yaml"))
    args = parser.parse_args()
    for path in analyze(args.results_dir, args.config).values():
        print(path, flush=True)


if __name__ == "__main__":
    main()
