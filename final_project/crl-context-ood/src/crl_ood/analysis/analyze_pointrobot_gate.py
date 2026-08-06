"""Result-only PointRobot gate analyzer; intentionally imports no RL stack."""

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

PRIMARY = ("train", "id")
OOD = ("ood_left", "ood_right")
TRAINING_GOALS = {"specialist_negative": -0.6, "specialist_center": 0.0, "specialist_positive": 0.6, "fixed_center": 0.0}
EXPECTED = {"contextual": {"hidden", "oracle"}, "specialist_negative": {"hidden"},
            "specialist_center": {"hidden"}, "specialist_positive": {"hidden"}, "fixed_center": {"oracle"}}
OUTPUT_NAMES = (
    "pointrobot_gate_seed_results.csv", "pointrobot_contextual_gaps_by_seed.csv",
    "pointrobot_train_id_summary.csv", "pointrobot_specialist_transfer_by_seed.csv",
    "pointrobot_specialist_summary.csv", "pointrobot_fixed_center_comparison.csv",
    "pointrobot_probe_summary.csv", "pointrobot_ood_descriptive.csv",
    "pointrobot_gate_findings.json", "pointrobot_gate_findings.md",
    "return_by_goal.png", "success_by_goal.png", "specialist_transfer_heatmap.png",
    "paired_oracle_gap.png", "probe_error_vs_history.png", "training_curves.png",
)


def analyze_pointrobot_gate(results_root: str | Path, config_path: str | Path,
                            output_dir: str | Path | None = None) -> dict[str, Path]:
    root, config_path = Path(results_root), Path(config_path)
    output = Path(output_dir) if output_dir else root / "analysis"
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    rows = _load_rows(root); _validate_matrix(rows)
    probe_rows = _read(root / "probe" / "probe_results_by_seed.csv")
    if not probe_rows:
        raise ValueError("Missing required probe_results_by_seed.csv")
    output.mkdir(parents=True, exist_ok=True)
    contextual = _contextual_gaps(rows)
    specialist_transfer = [row for row in rows if str(row["kind"]).startswith("specialist_") and row["split"] in PRIMARY]
    specialist_summary = _aggregate_specialists(specialist_transfer)
    fixed = _fixed_center(rows)
    train_id = _train_id_summary(rows)
    probe_summary = _probe_summary(probe_rows)
    ood = [dict(row, analysis_role="descriptive_only") for row in rows if row["split"] in OOD]
    findings = _gate(rows, contextual, fixed, probe_rows, config["gate"])
    paths = {name: output / name for name in OUTPUT_NAMES}
    for name, data in zip(OUTPUT_NAMES[:8], (rows, contextual, train_id, specialist_transfer, specialist_summary, fixed, probe_summary, ood), strict=True):
        _write(paths[name], data)
    paths[OUTPUT_NAMES[8]].write_text(json.dumps(findings, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths[OUTPUT_NAMES[9]].write_text(_markdown(findings), encoding="utf-8")
    _plots(root, rows, contextual, specialist_summary, probe_summary, output)
    return paths


def _load_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((root / "runs").glob("*/context_returns.csv")):
        for raw in _read(path):
            row: dict[str, Any] = dict(raw)
            for field in ("seed", "context_id", "episodes"): row[field] = int(raw[field])
            for field in ("goal_angle", "goal_cos", "goal_sin", "goal_x", "goal_y", "mean_return", "std_return", "success_rate", "mean_final_distance", "mean_minimum_distance"):
                row[field] = float(raw[field])
            row["training_goal_angle"] = TRAINING_GOALS.get(str(row["kind"]), "multiple")
            row["roles"] = "specialist;fixed_center_hidden" if row["kind"] == "specialist_center" else "specialist" if str(row["kind"]).startswith("specialist_") else "fixed_center_oracle" if row["kind"] == "fixed_center" else "contextual"
            rows.append(row)
    if not rows: raise ValueError("No PointRobot context results found")
    return sorted(rows, key=lambda x: (str(x["kind"]), str(x["method"]), int(x["seed"]), str(x["split"]), float(x["goal_angle"])))


def _validate_matrix(rows: list[dict[str, Any]]) -> None:
    actual = {(row["kind"], row["method"], row["seed"]) for row in rows}
    expected = {(kind, mode, seed) for kind, modes in EXPECTED.items() for mode in modes for seed in (0, 1)}
    if actual != expected: raise ValueError(f"Incomplete/unexpected PointRobot result matrix: {sorted(actual)}")
    expected_counts = {"train": 5, "id": 4, "ood_left": 2, "ood_right": 2}
    for combination in expected:
        subset = [x for x in rows if (x["kind"], x["method"], x["seed"]) == combination]
        counts = {split: sum(x["split"] == split for x in subset) for split in (*PRIMARY, *OOD)}
        if counts != expected_counts: raise ValueError(f"Invalid split rows for {combination}: {counts}")


def _contextual_gaps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for seed in (0, 1):
        for split in (*PRIMARY, *OOD):
            hidden = [x for x in rows if x["kind"] == "contextual" and x["method"] == "hidden" and x["seed"] == seed and x["split"] == split]
            oracle = [x for x in rows if x["kind"] == "contextual" and x["method"] == "oracle" and x["seed"] == seed and x["split"] == split]
            output.append({"seed": seed, "split": split,
                           "hidden_return": _mean(hidden, "mean_return"), "oracle_return": _mean(oracle, "mean_return"),
                           "oracle_minus_hidden_return": _mean(oracle, "mean_return") - _mean(hidden, "mean_return"),
                           "hidden_success_rate": _mean(hidden, "success_rate"), "oracle_success_rate": _mean(oracle, "success_rate"),
                           "oracle_minus_hidden_success_rate": _mean(oracle, "success_rate") - _mean(hidden, "success_rate"),
                           "analysis_role": "gate" if split in PRIMARY else "descriptive_only"})
    return output


def _train_id_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    keys = sorted({(x["kind"], x["method"], x["split"]) for x in rows if x["split"] in PRIMARY})
    for kind, method, split in keys:
        values = [x for x in rows if (x["kind"], x["method"], x["split"]) == (kind, method, split)]
        output.append({"kind": kind, "method": method, "split": split, "seed_context_rows": len(values),
                       "mean_return": _mean(values, "mean_return"), "mean_success_rate": _mean(values, "success_rate"),
                       "mean_final_distance": _mean(values, "mean_final_distance"), "confidence_interval": "not_computed_two_seeds"})
    return output


def _aggregate_specialists(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    keys = sorted({(x["training_goal_angle"], x["split"], x["goal_angle"]) for x in rows})
    for training, split, goal in keys:
        values = [x for x in rows if (x["training_goal_angle"], x["split"], x["goal_angle"]) == (training, split, goal)]
        output.append({"training_goal_angle": training, "evaluation_split": split, "evaluation_goal_angle": goal,
                       "seed_count": len(values), "mean_return": _mean(values, "mean_return"),
                       "mean_success_rate": _mean(values, "success_rate"), "mean_final_distance": _mean(values, "mean_final_distance")})
    return output


def _fixed_center(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output=[]
    for seed in (0,1):
        hidden=_one(rows,"specialist_center","hidden",seed,"train",0.0); oracle=_one(rows,"fixed_center","oracle",seed,"train",0.0)
        output.append({"seed":seed,"goal_angle":0.0,"hidden_run_id":hidden["run_id"],"hidden_reuse":"specialist_center",
                       "oracle_run_id":oracle["run_id"],"hidden_success_rate":hidden["success_rate"],"oracle_success_rate":oracle["success_rate"],
                       "absolute_success_rate_gap":abs(oracle["success_rate"]-hidden["success_rate"]),
                       "hidden_return":hidden["mean_return"],"oracle_return":oracle["mean_return"]})
    return output


def _probe_summary(rows: list[dict[str,str]]) -> list[dict[str,Any]]:
    output=[]
    keys=sorted({(x["probe"],int(x["history_length"]),x["split"]) for x in rows})
    for probe,h,split in keys:
        values=[float(x["circular_angle_mae"]) for x in rows if (x["probe"],int(x["history_length"]),x["split"])==(probe,h,split)]
        output.append({"probe":probe,"history_length":h,"split":split,"seed_count":len(values),"mean_circular_angle_mae":statistics.fmean(values),
                       "std_across_seeds_descriptive":statistics.pstdev(values),"confidence_interval":"not_computed_two_seeds","analysis_role":"gate" if split=="id" else "descriptive_only"})
    return output


def _gate(rows: list[dict[str,Any]], gaps: list[dict[str,Any]], fixed: list[dict[str,Any]], probe: list[dict[str,str]], gate: dict[str,Any]) -> dict[str,Any]:
    own=[]
    for seed in (0,1):
        for kind,goal in (("specialist_negative",-0.6),("specialist_center",0.0),("specialist_positive",0.6)):
            row=_one(rows,kind,"hidden",seed,"train",goal)
            own.append({"seed":seed,"kind":kind,"goal_angle":goal,"success_rate":row["success_rate"],"mean_final_distance":row["mean_final_distance"],
                        "success_pass":row["success_rate"]>=float(gate["specialist_min_own_success_rate"]),
                        "distance_pass":row["mean_final_distance"]<=float(gate["specialist_max_own_mean_final_distance"])})
    nearest=[]
    for seed in (0,1):
        wins=total=0; details=[]
        specialists=[x for x in rows if str(x["kind"]).startswith("specialist_") and x["seed"]==seed and x["split"] in PRIMARY]
        for split in PRIMARY:
            for goal in sorted({x["goal_angle"] for x in specialists if x["split"]==split}):
                candidates=[x for x in specialists if x["split"]==split and math.isclose(x["goal_angle"],goal,abs_tol=1e-12)]
                best=max(x["mean_return"] for x in candidates)
                winners={float(x["training_goal_angle"]) for x in candidates if math.isclose(x["mean_return"],best,abs_tol=1e-12)}
                distances={g:abs(g-goal) for g in (-0.6,0.0,0.6)}; minimum=min(distances.values())
                nearest_goals={g for g,d in distances.items() if math.isclose(d,minimum,abs_tol=1e-12)}
                passed=bool(winners & nearest_goals); wins+=int(passed); total+=1
                details.append({"split":split,"goal_angle":goal,"winning_specialists":sorted(winners),"nearest_specialists":sorted(nearest_goals),"pass":passed})
        fraction=wins/total; nearest.append({"seed":seed,"nearest_best_count":wins,"goal_count":total,"fraction":fraction,
                                             "pass":fraction>float(gate["specialist_nearest_majority_fraction"]),"details":details})
    contextual=[]
    for seed in (0,1):
        values=[x for x in gaps if x["seed"]==seed and x["split"] in PRIMARY]
        better=all(x["oracle_minus_hidden_return"]>0 and x["oracle_minus_hidden_success_rate"]>0 for x in values)
        gains={x["split"]:x["oracle_minus_hidden_success_rate"] for x in values}
        gain=any(v>=float(gate["contextual_min_oracle_success_gain"]) for v in gains.values()) and all(v>=0 for v in gains.values())
        contextual.append({"seed":seed,"splits":values,"higher_return_and_success_pass":better,"minimum_gain_and_no_loss_pass":gain})
    probe_support=[]
    for seed in (0,1):
        by={(x["probe"],int(x["history_length"])):float(x["circular_angle_mae"]) for x in probe if int(x["seed"])==seed and x["split"]=="id"}
        state=by[("state_only",0)]; h1=by[("history_h1",1)]
        long={h:by[(f"history_h{h}",h)] for h in map(int,gate["probe_long_history_candidates"])}
        reduction=max((state-v)/state for v in long.values()) if state else 0.0
        beyond=(h1-min(long.values()))/h1 if h1 else 0.0
        probe_support.append({"seed":seed,"state_only_id_mae":state,"history_h1_id_mae":h1,"long_history_id_mae":long,
                              "best_relative_reduction_vs_state":reduction,"relative_improvement_beyond_h1":beyond,
                              "long_history_pass":reduction>=float(gate["probe_min_relative_id_mae_reduction"]),
                              "beyond_h1_pass":beyond>=float(gate["probe_min_h1_relative_improvement"])})
    criteria={
        "specialist_own_goal_success":all(x["success_pass"] for x in own),
        "specialist_own_goal_final_distance":all(x["distance_pass"] for x in own),
        "specialist_goal_dependent_nearest_best":all(x["pass"] for x in nearest),
        "contextual_oracle_higher_return_and_success_train_id":all(x["higher_return_and_success_pass"] for x in contextual),
        "contextual_oracle_min_success_gain_no_other_loss":all(x["minimum_gain_and_no_loss_pass"] for x in contextual),
        "fixed_center_abs_success_gap":all(x["absolute_success_rate_gap"]<=float(gate["fixed_center_max_abs_success_gap"]) for x in fixed),
        "probe_long_history_relative_reduction":all(x["long_history_pass"] for x in probe_support),
        "probe_improves_beyond_h1":all(x["beyond_h1_pass"] for x in probe_support),
    }
    return {"accepted":all(criteria.values()),"criteria":criteria,"thresholds":gate,"supporting_values":{"specialist_own_goal":own,"specialist_nearest":nearest,"contextual":contextual,"fixed_center":fixed,"probe":probe_support},
            "primary_splits":["train","id"],"ood_excluded_from_acceptance":["ood_left","ood_right"],"confidence_intervals":"not_computed_two_seeds"}


def _markdown(findings: dict[str,Any]) -> str:
    lines=["# Dense Semi-Circle PointRobot gate findings","",f"Overall result-only gate: **{'ACCEPT' if findings['accepted'] else 'REJECT'}**","",
           "Two seeds are reported descriptively; no confidence intervals are computed. OOD-left and OOD-right are separate and excluded from acceptance.","","## Individual criteria",""]
    lines.extend(f"- {'PASS' if value else 'FAIL'}: `{key}`" for key,value in findings["criteria"].items())
    return "\n".join(lines)+"\n"


def _plots(root:Path, rows:list[dict[str,Any]], gaps:list[dict[str,Any]], specialist:list[dict[str,Any]], probe:list[dict[str,Any]], output:Path)->None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    contextual=[x for x in rows if x["kind"]=="contextual"]
    for metric,name,ylabel in (("mean_return","return_by_goal.png","Mean return"),("success_rate","success_by_goal.png","Success rate")):
        fig,ax=plt.subplots(figsize=(7,4))
        for method in ("hidden","oracle"):
            grouped=defaultdict(list)
            for x in contextual:
                if x["method"]==method: grouped[x["goal_angle"]].append(x[metric])
            xs=sorted(grouped); ax.plot(xs,[statistics.fmean(grouped[x]) for x in xs],marker="o",label=method)
        ax.set(xlabel="Goal angle (rad)",ylabel=ylabel,title=f"Contextual {ylabel.lower()} by goal"); ax.legend(); fig.tight_layout(); fig.savefig(output/name,dpi=160); plt.close(fig)
    trains=(-0.6,0.0,0.6); goals=sorted({x["evaluation_goal_angle"] for x in specialist}); lookup={(x["training_goal_angle"],x["evaluation_goal_angle"]):x["mean_success_rate"] for x in specialist}
    fig,ax=plt.subplots(figsize=(8,3.5)); image=ax.imshow([[lookup[(t,g)] for g in goals] for t in trains],aspect="auto",cmap="viridis",vmin=0,vmax=1)
    ax.set(xticks=range(len(goals)),xticklabels=goals,yticks=range(3),yticklabels=trains,xlabel="Evaluation goal",ylabel="Training goal",title="Specialist success transfer"); fig.colorbar(image,ax=ax); fig.tight_layout(); fig.savefig(output/"specialist_transfer_heatmap.png",dpi=160); plt.close(fig)
    primary=[x for x in gaps if x["split"] in PRIMARY]; fig,ax=plt.subplots(figsize=(6,4)); labels=[f"s{x['seed']} {x['split']}" for x in primary]
    ax.bar(labels,[x["oracle_minus_hidden_success_rate"] for x in primary]); ax.axhline(0,color="black",linewidth=.8); ax.set(ylabel="Oracle - hidden success",title="Paired contextual oracle gap"); fig.tight_layout(); fig.savefig(output/"paired_oracle_gap.png",dpi=160); plt.close(fig)
    idp=sorted([x for x in probe if x["split"]=="id"],key=lambda x:x["history_length"]); fig,ax=plt.subplots(figsize=(6,4)); ax.plot([x["history_length"] for x in idp],[x["mean_circular_angle_mae"] for x in idp],marker="o"); ax.set(xlabel="History length (0 = state only)",ylabel="ID circular-angle MAE",title="Probe error vs history"); fig.tight_layout(); fig.savefig(output/"probe_error_vs_history.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for path in sorted((root/"runs").glob("*/training_metrics.csv")):
        data=_read(path)
        if data: ax.plot([float(x["environment_steps"]) for x in data],[float(x["episode_return"]) for x in data],alpha=.45)
    ax.set(xlabel="Environment steps",ylabel="Episode return",title="PointRobot training curves"); fig.tight_layout(); fig.savefig(output/"training_curves.png",dpi=160); plt.close(fig)


def _one(rows: list[dict[str,Any]],kind:str,method:str,seed:int,split:str,goal:float)->dict[str,Any]:
    found=[x for x in rows if x["kind"]==kind and x["method"]==method and x["seed"]==seed and x["split"]==split and math.isclose(x["goal_angle"],goal,abs_tol=1e-12)]
    if len(found)!=1: raise ValueError(f"Expected one result for {kind}/{method}/s{seed}/{split}/{goal}")
    return found[0]


def _mean(rows:list[dict[str,Any]],field:str)->float: return statistics.fmean(float(x[field]) for x in rows)
def _read(path:Path)->list[dict[str,str]]:
    if not path.is_file(): return []
    with path.open(encoding="utf-8",newline="") as handle: return list(csv.DictReader(handle))
def _write(path:Path,rows:list[dict[str,Any]])->None:
    if not rows: raise ValueError(f"Cannot write empty {path.name}")
    with path.open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=tuple(rows[0])); writer.writeheader(); writer.writerows(rows)


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--results-root",type=Path,default=Path("results/pointrobot_gate")); parser.add_argument("--config",type=Path,default=Path("configs/pointrobot_gate/gate.yaml")); parser.add_argument("--output-dir",type=Path); args=parser.parse_args()
    for path in analyze_pointrobot_gate(args.results_root,args.config,args.output_dir).values(): print(path,flush=True)
if __name__=="__main__": main()
