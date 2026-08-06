from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from crl_ood.analysis.analyze_pointrobot_gate import OUTPUT_NAMES, analyze_pointrobot_gate

ROOT=Path(__file__).parents[1]; CONFIG=ROOT/"configs/pointrobot_gate/gate.yaml"
SPLITS={"train":[-.6,-.3,0,.3,.6],"id":[-.45,-.15,.15,.45],"ood_left":[-1,-.8],"ood_right":[.8,1]}
RUNS=(("contextual","hidden",None),("contextual","oracle",None),("specialist_negative","hidden",-.6),("specialist_center","hidden",0.),("specialist_positive","hidden",.6),("fixed_center","oracle",0.))


def test_result_only_analyzer_dependency_isolation():
    code=("import sys; blocked={'carl','gym','gymnasium','stable_baselines3','torch'}; sys.meta_path.insert(0,type('B',(),{'find_spec':lambda s,n,p=None,t=None: (_ for _ in ()).throw(RuntimeError(n)) if n.split('.')[0] in blocked else None})()); import crl_ood.analysis.analyze_pointrobot_gate; print('isolated')")
    completed=subprocess.run([sys.executable,"-c",code],check=True,capture_output=True,text=True)
    assert completed.stdout.strip()=="isolated"


def test_synthetic_gate_pass_writes_every_required_output(tmp_path):
    root=_fixture(tmp_path/"pointrobot_gate")
    paths=analyze_pointrobot_gate(root,CONFIG)
    assert set(paths)==set(OUTPUT_NAMES)
    assert all(path.is_file() and path.stat().st_size>0 for path in paths.values())
    findings=json.loads(paths["pointrobot_gate_findings.json"].read_text())
    assert findings["accepted"] is True; assert all(findings["criteria"].values())
    assert findings["ood_excluded_from_acceptance"]==["ood_left","ood_right"]
    ood=_read(paths["pointrobot_ood_descriptive.csv"])
    assert {x["split"] for x in ood}=={"ood_left","ood_right"}; assert {x["analysis_role"] for x in ood}=={"descriptive_only"}
    assert "no confidence intervals" in paths["pointrobot_gate_findings.md"].read_text().lower()


def test_synthetic_gate_failure_fixtures_and_ood_noninterference(tmp_path):
    root=_fixture(tmp_path/"pointrobot_gate")
    hidden=next((root/"runs").glob("specialist_negative__hidden__seed_0/context_returns.csv")); rows=_read(hidden)
    for row in rows:
        if row["split"]=="train" and float(row["goal_angle"])==-.6: row["success_rate"]="0.1"
        if row["split"]=="ood_left": row["success_rate"]="0.0"; row["mean_return"]="-99999"
    _write(hidden,rows)
    findings=json.loads(analyze_pointrobot_gate(root,CONFIG,root/"failed_analysis")["pointrobot_gate_findings.json"].read_text())
    assert findings["accepted"] is False; assert findings["criteria"]["specialist_own_goal_success"] is False
    # A fresh fixture with catastrophic OOD alone still passes.
    clean=_fixture(tmp_path/"ood_only")
    for path in (clean/"runs").glob("*/context_returns.csv"):
        data=_read(path)
        for row in data:
            if row["split"].startswith("ood_"): row["success_rate"]="0"; row["mean_return"]="-99999"
        _write(path,data)
    assert json.loads(analyze_pointrobot_gate(clean,CONFIG)["pointrobot_gate_findings.json"].read_text())["accepted"] is True


def _fixture(root:Path)->Path:
    import math
    for kind,method,training in RUNS:
        for seed in (0,1):
            run_id=f"{kind}__{method}__seed_{seed}"; directory=root/"runs"/run_id; directory.mkdir(parents=True)
            rows=[]
            for split,goals in SPLITS.items():
                for context_id,goal in enumerate(goals):
                    if kind=="contextual": success=.5 if method=="hidden" else .8; value=-30-abs(goal)+(5 if method=="oracle" else 0); distance=.2 if method=="hidden" else .08
                    elif kind.startswith("specialist_"):
                        delta=abs(goal-float(training)); success=max(0,.9-delta); value=-100*delta; distance=.05+delta
                    else: success=.85; value=-1.; distance=.07
                    rows.append({"run_id":run_id,"method":method,"kind":kind,"seed":seed,"split":split,"context_id":context_id,
                                 "goal_angle":goal,"goal_cos":math.cos(goal),"goal_sin":math.sin(goal),"goal_x":math.cos(goal),"goal_y":math.sin(goal),
                                 "episodes":10,"mean_return":value,"std_return":1.,"success_rate":success,"mean_final_distance":distance,"mean_minimum_distance":distance})
            _write(directory/"context_returns.csv",rows)
            _write(directory/"training_metrics.csv",[{"environment_steps":50,"episode_return":-30},{"environment_steps":100,"episode_return":-20}])
    probe_dir=root/"probe"; probe_dir.mkdir()
    maes={"state_only":(0,1.0),"history_h1":(1,.7),"history_h3":(3,.55),"history_h5":(5,.4),"history_h10":(10,.3)}
    probe=[]
    for seed in (0,1):
        for name,(history,mae) in maes.items():
            for split in ("id","ood_left","ood_right"):
                probe.append({"seed":seed,"probe":name,"history_length":history,"split":split,"circular_angle_mae":mae,"cos_mae":mae,"sin_mae":mae,"analysis_role":"gate" if split=="id" else "descriptive_only"})
    _write(probe_dir/"probe_results_by_seed.csv",probe)
    return root


def _read(path:Path)->list[dict[str,str]]:
    with path.open(encoding="utf-8",newline="") as handle:return list(csv.DictReader(handle))
def _write(path:Path,rows:list[dict[str,object]])->None:
    with path.open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=tuple(rows[0])); writer.writeheader(); writer.writerows(rows)

