from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from crl_ood.pointrobot_encoders.dataset import (ENCODER_INPUT_FIELDS, TRAIN_CONTEXTS, WindowIndex,
    behavior_actions, collect_arrays, dataset_checksum, load_spec, make_window, pointrobot_reward,
    save_dataset, transition_features, window_indices)
from crl_ood.pointrobot_encoders.downstream import build_jobs
from crl_ood.pointrobot_encoders.models import (ContrastiveHistoryEncoder, VAEHistoryEncoder,
    checkpoint_payload, contrastive_objective, parameter_counts, vae_objective)
from crl_ood.pointrobot_encoders.training import hard_negative_rewards, load_frozen_checkpoint
from crl_ood.pointrobot_encoders.wrapper import FrozenHistoryObservation, make_policy_env
from crl_ood.pointrobot_gate.environment import DenseSemiCirclePointRobot

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_spec(ROOT / "configs/pointrobot_encoders/primary.yaml")


@pytest.fixture(scope="module")
def tiny():
    return collect_arrays(CONFIG, "tiny")


def test_deterministic_dataset_and_checksum_stability(tiny):
    arrays, metadata = tiny
    other_arrays, other_metadata = collect_arrays(CONFIG, "tiny")
    assert all(np.array_equal(value, other_arrays[key]) for key, value in arrays.items())
    assert metadata["dataset_checksum"] == other_metadata["dataset_checksum"]
    assert dataset_checksum(arrays, metadata) == metadata["dataset_checksum"]


def test_identical_actions_across_matched_contexts(tiny):
    arrays, _ = tiny
    per_context = len(arrays["episode_ids"]) // len(TRAIN_CONTEXTS)
    for trajectory in range(per_context):
        matched = arrays["actions"][trajectory::per_context]
        assert all(np.array_equal(matched[0], item) for item in matched[1:])


def test_behavior_prefix_and_random_variant():
    prefix = CONFIG["dataset"]["orthogonal_prefix"]
    selected = behavior_actions("orthogonal_then_isotropic", 8, 4, prefix)
    random = behavior_actions("random_only", 8, 4, prefix)
    assert np.array_equal(selected[:4], np.asarray(prefix, dtype=np.float32))
    assert not np.array_equal(selected[:4], random[:4])


def test_no_context_input_leakage():
    assert ENCODER_INPUT_FIELDS == ("state", "action", "reward", "next_state")
    assert "context" not in ENCODER_INPUT_FIELDS


def test_episode_split_isolation(tiny):
    arrays, _ = tiny
    train = {x.episode for x in window_indices(arrays, "train", 5, 5)}
    validation = {x.episode for x in window_indices(arrays, "validation", 5, 5)}
    assert train and validation and train.isdisjoint(validation)


def test_history_indexing_and_no_future_leakage(tiny):
    arrays, _ = tiny
    window = make_window(arrays, WindowIndex(0, 7), 5, 5)
    expected = transition_features(arrays["states"][0, 2:7], arrays["actions"][0, 2:7],
        arrays["rewards"][0, 2:7], arrays["next_states"][0, 2:7])
    normalized = (expected - arrays["normalization_mean"]) / arrays["normalization_std"]
    assert np.allclose(window["history"], normalized)
    assert np.array_equal(window["future_actions"], arrays["actions"][0, 7:12])
    assert window["timestep"] == 7


def test_reward_state_alignment_and_targets(tiny):
    arrays, _ = tiny
    expected = pointrobot_reward(arrays["next_states"][0], arrays["actions"][0], float(arrays["contexts"][0]))
    assert np.allclose(expected, arrays["rewards"][0], atol=1e-6)
    window = make_window(arrays, WindowIndex(0, 3), 5, 5)
    assert np.allclose(window["future_state_deltas"], np.diff(arrays["states"][0, 3:9], axis=0))
    assert np.array_equal(window["future_rewards"], arrays["rewards"][0, 3:8])


def test_padding_masking_and_empty_history(tiny):
    arrays, _ = tiny
    empty = make_window(arrays, WindowIndex(0, 0), 5, 5)
    partial = make_window(arrays, WindowIndex(0, 2), 5, 5)
    assert empty["length"] == 0 and not empty["mask"].any() and not empty["history"].any()
    assert partial["length"] == 2 and partial["mask"].tolist() == [True, True, False, False, False]


def _batch(batch=4):
    return {"history": torch.randn(batch, 5, 7), "length": torch.full((batch,), 5),
        "mask": torch.ones(batch, 5, dtype=torch.bool), "current_state": torch.randn(batch, 2),
        "future_actions": torch.randn(batch, 5, 2), "future_state_deltas": torch.randn(batch, 5, 2),
        "future_states": torch.randn(batch, 5, 2), "future_rewards": torch.randn(batch, 5),
        "context": torch.tensor(list(TRAIN_CONTEXTS[:batch])), "episode_id": torch.arange(batch),
        "timestep": torch.arange(batch)}


def test_identical_backbone_architecture_latent_and_parameter_reporting():
    vae, contrastive = VAEHistoryEncoder(), ContrastiveHistoryEncoder()
    assert repr(vae.backbone) == repr(contrastive.backbone)
    assert vae.backbone.latent_dim == contrastive.backbone.latent_dim == 8
    counts_v, counts_c = parameter_counts(vae), parameter_counts(contrastive)
    assert counts_v["backbone"] == counts_c["backbone"]
    assert counts_v["total_training"] == counts_v["backbone"] + counts_v["method_specific"]


def test_vae_reparameterization_mean_and_kl():
    model, batch = VAEHistoryEncoder(), _batch()
    deterministic_a = model.encode(batch["history"], batch["length"], batch["mask"], True)
    deterministic_b = model.encode(batch["history"], batch["length"], batch["mask"], True)
    stochastic_a = model.encode(batch["history"], batch["length"], batch["mask"], False)
    stochastic_b = model.encode(batch["history"], batch["length"], batch["mask"], False)
    assert torch.equal(deterministic_a, deterministic_b)
    assert not torch.equal(stochastic_a, stochastic_b)
    output = model(batch); loss = vae_objective(output, batch, 1, 1, .001)
    assert set(loss) == {"total", "state_reconstruction", "reward_reconstruction", "kl"}
    assert loss["kl"] >= 0 and output["predicted_state_deltas"].shape == (4, 5, 2)
    assert output["predicted_rewards"].shape == (4, 5)


def test_contrastive_positive_and_infonce_validity():
    model, batch = ContrastiveHistoryEncoder(), _batch()
    negative = batch["future_rewards"] + 1
    losses = contrastive_objective(model, batch, negative, .1)
    assert losses["logits"].shape == (4, 2) and torch.isfinite(losses["total"])
    in_batch = contrastive_objective(model, batch, negative, .1, "in_batch")
    assert in_batch["logits"].shape == (4, 4)
    with pytest.raises(ValueError):
        contrastive_objective(model, {k: v[:1] for k, v in batch.items()}, negative[:1], .1, "in_batch")


def test_hard_negative_same_state_action_different_training_goal():
    batch = _batch()
    original_states, original_actions = batch["future_states"].clone(), batch["future_actions"].clone()
    negative, provenance = hard_negative_rewards(batch)
    assert torch.equal(original_states, batch["future_states"])
    assert torch.equal(original_actions, batch["future_actions"])
    assert all(row["positive_goal"] != row["negative_goal"] for row in provenance)
    assert all(row["negative_goal"] in TRAIN_CONTEXTS for row in provenance)
    assert all(row["reward_targets_different"] for row in provenance)
    assert not torch.allclose(negative, batch["future_rewards"])


def _checkpoint(tmp_path: Path, method: str) -> tuple[Path, str]:
    model = VAEHistoryEncoder() if method == "vae" else ContrastiveHistoryEncoder()
    payload = checkpoint_payload(model, method, CONFIG, {"mean": [0] * 7, "std": [1] * 7}, "dataset", 0, 1, 1.0)
    path = tmp_path / f"{method}.pt"; torch.save(payload, path)
    return path, "dataset"


def test_checkpoint_reload_determinism_and_frozen_gradients(tmp_path):
    checkpoint, checksum = _checkpoint(tmp_path, "vae")
    first, _ = load_frozen_checkpoint(checkpoint, "vae", checksum)
    second, _ = load_frozen_checkpoint(checkpoint, "vae", checksum)
    batch = _batch()
    assert torch.equal(first.encode(batch["history"], batch["length"], batch["mask"]),
                       second.encode(batch["history"], batch["length"], batch["mask"]))
    assert not any(parameter.requires_grad for parameter in first.parameters())


def test_reset_clears_history_empty_parity_and_observation_dimensions(tmp_path):
    kwargs = {"horizon": 5}
    checkpoint_v, checksum = _checkpoint(tmp_path, "vae")
    checkpoint_c, _ = _checkpoint(tmp_path, "contrastive")
    vae = FrozenHistoryObservation(DenseSemiCirclePointRobot(0.0, horizon=5), checkpoint_v, "vae", checksum)
    contrastive = FrozenHistoryObservation(DenseSemiCirclePointRobot(0.0, horizon=5), checkpoint_c, "contrastive", checksum)
    obs_v, _ = vae.reset(); obs_c, _ = contrastive.reset()
    assert obs_v.shape == obs_c.shape == (10,) and np.array_equal(obs_v[2:], obs_c[2:])
    stepped, *_ = vae.step(np.array([1, 0], dtype=np.float32)); assert len(vae._history) == 1
    reset, _ = vae.reset(); assert len(vae._history) == 0 and np.array_equal(reset[2:], obs_v[2:])
    assert make_policy_env("no_context", 0.0, kwargs).observation_space.shape == (2,)
    assert make_policy_env("oracle", 0.0, kwargs).observation_space.shape == (4,)


def test_explicit_checkpoint_provenance_and_full_matrix_lock(tmp_path):
    downstream = yaml.safe_load((ROOT / "configs/pointrobot_encoders/downstream.yaml").read_text())
    checkpoints = {"vae": str(tmp_path / "v.pt"), "contrastive": str(tmp_path / "c.pt")}
    jobs = build_jobs(downstream, "integration_pilot", checkpoints, "checksum")
    assert len(jobs) == 8 and all(job.checkpoint and job.dataset_checksum for job in jobs if job.method in {"vae", "contrastive"})
    with pytest.raises(ValueError): build_jobs(downstream, "integration_pilot", {}, None)
    with pytest.raises(RuntimeError): build_jobs(downstream, "full_primary", checkpoints, "checksum")


def test_overwrite_and_partial_output_protection(tmp_path, tiny):
    arrays, metadata = tiny
    target = tmp_path / "dataset"; target.mkdir(); (target / "partial").write_text("x")
    with pytest.raises(FileExistsError): save_dataset(target, arrays, metadata)
    with pytest.raises(FileExistsError): save_dataset(target, arrays, metadata, overwrite=True)


def test_result_only_analyzer_dependency_isolation():
    source = (ROOT / "src/crl_ood/analysis/analyze_pointrobot_encoders.py").read_text()
    tree = ast.parse(source)
    imports = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imports |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert imports.isdisjoint({"gym", "gymnasium", "carl", "stable_baselines3", "torch"})


def test_every_protected_file_is_preserved():
    manifest = ROOT / "results/pointrobot_encoders/protected_before.sha256"
    assert manifest.is_file()
    for line in manifest.read_text().splitlines():
        expected, relative = line.split(None, 1)
        path = ROOT / relative.strip()
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
