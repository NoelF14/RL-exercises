from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from crl_ood.analysis.analyze_pointrobot_probe_audit import evaluate_v2
from crl_ood.pointrobot_probe_audit.core import (
    TRANSITION_FIELDS, action_sequence, analytic_goal, circular_angle_error,
    engineered_features, normalize_prediction, raw_history_features,
    reward_equation_b, sequence_features, transition_history,
    validate_no_target_leakage, validate_split_isolation, validate_transition_alignment,
)
from crl_ood.pointrobot_probe_audit.run import collect_dataset

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/pointrobot_probe_audit/audit.yaml"


def _transition(position: tuple[float, float], goal: np.ndarray, action=(0.0, 0.0), penalty=0.01):
    p = np.asarray(position, dtype=float); a = np.asarray(action, dtype=float)
    reward = -(np.sum((p - goal) ** 2) + penalty * np.sum(a**2))
    return {"state": np.zeros(2), "action": a, "reward": reward, "next_state": p}


def test_exact_reward_equation_rearrangement():
    goal = np.array([math.cos(.4), math.sin(.4)])
    item = _transition((.23, -.41), goal, (.3, -.7))
    assert reward_equation_b(item["next_state"], item["action"], item["reward"], .01) == pytest.approx(item["next_state"] @ goal)


def test_analytic_recovery_from_two_non_collinear_positions():
    goal = np.array([math.cos(-.37), math.sin(-.37)])
    result = analytic_goal([_transition((.1, 0), goal), _transition((.1, .1), goal)], .01)
    assert result["rank"] == 2 and result["geometrically_identifiable"]
    assert result["estimated_goal_cos"] == pytest.approx(goal[0])
    assert result["estimated_goal_sin"] == pytest.approx(goal[1])


def test_rank_one_is_ambiguous_and_condition_is_infinite():
    goal = np.array([1.0, 0.0])
    result = analytic_goal([_transition((.1, 0), goal), _transition((.2, 0), goal)], .01)
    assert result["rank"] == 1 and not result["geometrically_identifiable"]
    assert math.isinf(result["condition_number"])


def test_prediction_normalization_and_circular_angle_error():
    np.testing.assert_allclose(normalize_prediction(np.array([3.0, 4.0])), [.6, .8])
    np.testing.assert_allclose(normalize_prediction(np.zeros(2)), [1.0, 0.0])
    assert circular_angle_error(-math.pi + .1, math.pi - .1) == pytest.approx(.2)


def test_deterministic_matched_actions_and_orthogonal_geometry():
    actions_a = action_sequence("deterministic_orthogonal", 50, 1)
    actions_b = action_sequence("deterministic_orthogonal", 50, 999)
    np.testing.assert_array_equal(actions_a, actions_b)
    positions = np.cumsum(.1 * actions_a, axis=0)
    assert np.linalg.matrix_rank(positions[-2:]) == 2


def test_transition_reward_alignment_and_off_by_one_negative_control():
    with CONFIG.open(encoding="utf-8") as handle: config = yaml.safe_load(handle)
    with (ROOT / config["experiment"]["source_config"]).open(encoding="utf-8") as handle: source = yaml.safe_load(handle)
    row = collect_dataset(source, {**config["audit"], "trajectories_per_context": 1}, "existing_exploratory", 0)[0]
    assert validate_transition_alignment(row, source["environment"]["action_penalty"])["transition_count"] == 50
    shifted = copy.deepcopy(row)
    shifted["rewards"] = np.roll(shifted["rewards"], 1)
    with pytest.raises(ValueError, match="resulting position"):
        validate_transition_alignment(shifted, source["environment"]["action_penalty"])


def test_histories_have_no_future_and_target_leakage_is_rejected():
    row = {"states": np.arange(12).reshape(6, 2), "actions": np.arange(10).reshape(5, 2), "rewards": np.arange(5)}
    history = transition_history(row, 2)
    np.testing.assert_array_equal(history[0]["state"], row["states"][3])
    np.testing.assert_array_equal(history[-1]["next_state"], row["states"][5])
    validate_no_target_leakage(set(TRANSITION_FIELDS))
    with pytest.raises(ValueError, match="Target leakage"):
        validate_no_target_leakage({"state", "context_id"})


def test_probe_inputs_are_invariant_to_target_and_split_metadata():
    row = {"states": np.array([[0., 0.], [.1, 0.], [.1, .1]]),
           "actions": np.array([[1., 0.], [0., 1.]]), "rewards": np.array([-1., -.8]),
           "goal_angle": .2, "goal_cos": .98, "goal_sin": .2, "context_id": 0, "split": "train"}
    altered = dict(row, goal_angle=2.0, goal_cos=-1.0, goal_sin=0.0, context_id=99, split="ood_left", filename="goal_2.csv")
    np.testing.assert_array_equal(raw_history_features(row, 2), raw_history_features(altered, 2))
    np.testing.assert_array_equal(sequence_features(row, 2), sequence_features(altered, 2))
    np.testing.assert_array_equal(engineered_features(row, 2, .01), engineered_features(altered, 2, .01))


def test_train_id_split_isolation():
    validate_split_isolation([{"split": "train"}, {"split": "train"}])
    with pytest.raises(ValueError, match="train only"):
        validate_split_isolation([{"split": "train"}, {"split": "id"}])


def test_all_probe_models_use_same_dataset_rows():
    goal = np.array([math.cos(.2), math.sin(.2)])
    actions = np.array([[1., 0.], [0., 1.]])
    states = np.array([[0., 0.], [.1, 0.], [.1, .1]])
    rewards = np.array([_transition(tuple(states[i + 1]), goal, tuple(actions[i]))["reward"] for i in range(2)])
    row = {"states": states, "actions": actions, "rewards": rewards}
    assert raw_history_features(row, 2).size == 12
    assert engineered_features(row, 2, .01).size == 5
    assert sequence_features(row, 2).shape == (2, 7)
    assert len(transition_history(row, 2)) == 2


def _synthetic_inputs(passing: bool):
    with CONFIG.open(encoding="utf-8") as handle: config = yaml.safe_load(handle)
    geometry = [{"behavior_policy": "deterministic_orthogonal", "history_length": 2, "split": "id",
                 "full_rank_fraction": 1.0 if passing else .9, "mean_circular_angle_error": .001 if passing else .1}]
    models = [{"model": "state_only", "history_length": 0, "split": "id", "mean_circular_angle_mae": .3}]
    for model in ("mlp", "gru"):
        models.extend([
            {"model": model, "history_length": 1, "split": "id", "mean_circular_angle_mae": .25},
            {"model": model, "history_length": 5, "split": "id", "mean_circular_angle_mae": .05 if passing else .24},
            {"model": model, "history_length": 10, "split": "id", "mean_circular_angle_mae": .05 if passing else .24},
        ])
    alignment = {key: passing for key in (
        "reward_alignment_pass", "no_future_transitions_pass", "target_constant_within_episode_pass",
        "forbidden_input_fields_absent_pass", "matched_actions_across_contexts_pass",
        "train_id_ood_fit_isolation_pass", "dataset_parity_pass")}
    return geometry, models, alignment, config


def test_synthetic_v2_pass_and_failure_fixtures():
    good = _synthetic_inputs(True)
    bad = _synthetic_inputs(False)
    assert evaluate_v2(*good, {"accepted": False})["accepted"]
    assert not evaluate_v2(*bad, {"accepted": False})["accepted"]


def test_result_only_analyzer_dependency_isolation():
    path = ROOT / "src/crl_ood/analysis/analyze_pointrobot_probe_audit.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
               for alias in node.names}
    assert imports.isdisjoint({"gym", "gymnasium", "carl", "stable_baselines3", "torch"})
    code = "import sys; blocked={'gym','gymnasium','carl','stable_baselines3','torch'}; sys.meta_path.insert(0,type('B',(),{'find_spec':lambda s,n,p=None,t=None: (_ for _ in ()).throw(RuntimeError(n)) if n.split('.')[0] in blocked else None})()); import crl_ood.analysis.analyze_pointrobot_probe_audit"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_every_protected_file_matches_probe_audit_manifests():
    before = ROOT / "results/pointrobot_probe_audit/protected_before.sha256"
    after = ROOT / "results/pointrobot_probe_audit/protected_after.sha256"
    assert before.read_bytes() == after.read_bytes()
    expected = {path: digest for digest, path in (line.split("  ", 1) for line in before.read_text().splitlines())}
    actual = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in expected}
    assert actual == expected
