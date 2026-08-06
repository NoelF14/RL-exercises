from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from crl_ood.pointrobot_gate.matrix import RunState, build_matrix, inspect_run
from crl_ood.pointrobot_gate.probe import collect_probe_dataset, history_features, run_probe, state_only_features
from crl_ood.pointrobot_gate.run import build_evaluation_plan, train_one
from crl_ood.pointrobot_gate.spec import EXPECTED_SPLITS, circular_distance, context_splits

ROOT=Path(__file__).parents[1]; MATRIX=ROOT/"configs/pointrobot_gate/matrix.yaml"


def test_splits_disjoint_and_ood_sides_separate():
    splits=context_splits(build_matrix(MATRIX)[0].config)
    assert {key:tuple(value.values()) for key,value in splits.items()}==EXPECTED_SPLITS
    assert set(splits["ood_left"].values()).isdisjoint(splits["ood_right"].values())
    assert max(splits["ood_left"].values())<min(splits["train"].values())<max(splits["train"].values())<min(splits["ood_right"].values())


def test_complete_12_job_matrix_center_dedup_and_200k_budget():
    jobs=build_matrix(MATRIX)
    assert len(jobs)==len({x.job_id for x in jobs})==len({x.output_dir for x in jobs})==12
    assert all(x.total_timesteps==200_000 for x in jobs)
    assert sum(x.kind=="contextual" for x in jobs)==4
    assert sum(x.kind.startswith("specialist_") for x in jobs)==6
    assert sum(x.kind=="fixed_center" for x in jobs)==2
    assert {x.kind for x in jobs if "fixed_center_hidden" in x.roles}=={"specialist_center"}
    assert not any(x.kind=="fixed_center" and x.mode=="hidden" for x in jobs)


def test_paired_evaluation_seeds_deterministic():
    jobs=build_matrix(MATRIX); hidden=next(x for x in jobs if x.kind=="contextual" and x.mode=="hidden" and x.seed==0); oracle=next(x for x in jobs if x.kind=="contextual" and x.mode=="oracle" and x.seed==0)
    hp,op=build_evaluation_plan(hidden),build_evaluation_plan(oracle)
    fields=("seed","split","context_id","goal_angle","episode_index","episode_seed")
    assert [tuple(x[k] for k in fields) for x in hp]==[tuple(x[k] for k in fields) for x in op]
    assert hp==build_evaluation_plan(hidden); assert len({x["episode_seed"] for x in hp})==len(hp)


def test_dry_run_dependency_isolated():
    code=("import sys; blocked={'carl','gym','gymnasium','stable_baselines3','torch'}; sys.meta_path.insert(0,type('B',(),{'find_spec':lambda s,n,p=None,t=None: (_ for _ in ()).throw(RuntimeError(n)) if n.split('.')[0] in blocked else None})()); from crl_ood.pointrobot_gate.run import main; "+f"sys.argv=['run','--matrix-config',{str(MATRIX)!r},'--dry-run']; main()")
    completed=subprocess.run([sys.executable,"-c",code],check=True,capture_output=True,text=True)
    assert "jobs=12 unique_atomic_runs=12 concurrency=1" in completed.stdout; assert completed.stdout.count("200000")==12


def test_pending_partial_and_overwrite_refusal(tmp_path):
    source=build_matrix(MATRIX)[0]; job=copy.copy(source); object.__setattr__(job,"output_dir",tmp_path/source.job_id)
    assert inspect_run(job).state is RunState.PENDING
    job.output_dir.mkdir(); (job.output_dir/"sentinel").write_text("keep")
    assert inspect_run(job).state is RunState.PARTIAL
    with pytest.raises(FileExistsError): train_one(job)
    assert (job.output_dir/"sentinel").read_text()=="keep"


def test_probe_feature_isolation_and_circular_error():
    trajectory={"states":np.array([[0,0],[.1,.2],[.2,.3]]),"actions":np.array([[1,0],[0,1]]),"rewards":np.array([-1.,-.5]),"goal_angle":.6}
    np.testing.assert_array_equal(state_only_features(trajectory),[.2,.3])
    history=history_features(trajectory,2)
    np.testing.assert_array_equal(history,np.r_[trajectory["states"].reshape(-1),trajectory["actions"].reshape(-1),trajectory["rewards"]])
    assert circular_distance(-np.pi+.1,np.pi-.1)==pytest.approx(.2)


def test_probe_dataset_uses_identical_actions_across_contexts_and_writes_outputs(tmp_path):
    config=copy.deepcopy(build_matrix(MATRIX)[0].config); config["probe"]["trajectories_per_context"]=2
    dataset=collect_probe_dataset(config,0)
    paired=[row for row in dataset if row["trajectory_index"]==0]
    assert len(paired)==13
    for row in paired[1:]:
        np.testing.assert_array_equal(row["actions"],paired[0]["actions"])
        np.testing.assert_array_equal(row["states"],paired[0]["states"])
    assert len({tuple(row["rewards"]) for row in paired})>1
    config_path=tmp_path/"probe.yaml"; config_path.write_text(yaml.safe_dump(config),encoding="utf-8")
    paths=run_probe(config_path,tmp_path/"probe")
    assert all(path.is_file() and path.stat().st_size>0 for path in paths.values())


def test_tiny_ppo_and_evaluation_smoke(tmp_path):
    source=build_matrix(MATRIX)[0]; config=copy.deepcopy(source.config)
    config["training"].update({"total_timesteps":128,"n_steps":64,"batch_size":32,"n_epochs":1,"device":"cpu"}); config["evaluation"]["episodes_per_context"]=1
    job=copy.copy(source); object.__setattr__(job,"config",config); object.__setattr__(job,"total_timesteps",128); object.__setattr__(job,"output_dir",tmp_path/source.job_id)
    run=train_one(job); assert inspect_run(job).state is RunState.COMPLETE
    assert (run/"model.zip").is_file(); assert (run/"success_metrics.csv").is_file(); assert (run/"distance_metrics.csv").is_file()
    assert (run/"episode_returns.csv").read_text().count("\n")==14
