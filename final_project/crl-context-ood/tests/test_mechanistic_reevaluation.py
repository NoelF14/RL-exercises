from __future__ import annotations

import numpy as np
import pytest

from crl_ood.mechanistic_audit.reevaluate import _rollouts, _summary, evaluation_seeds


class _Policy:
    def predict(self, observation, deterministic):
        assert deterministic is True
        return np.array([0.0], dtype=np.float32), None


class _Env:
    def __init__(self): self.steps = 0
    def reset(self, seed): self.steps = 0; return np.array([seed], dtype=np.float32), {}
    def step(self, action):
        self.steps += 1
        return np.array([0.0]), 1.5, False, self.steps == 2, {}
    def close(self): pass


def test_persisted_seed_plan_is_new_deterministic_and_exactly_100():
    seeds = evaluation_seeds()
    assert len(seeds) == len(set(seeds)) == 100
    assert min(seeds) >= 5_000_000
    with pytest.raises(ValueError, match="exactly 100"):
        evaluation_seeds(5)


def test_evaluation_rollout_smoke_never_trains(tmp_path):
    checkpoint = tmp_path / "model.zip"
    checkpoint.write_bytes(b"existing-checkpoint")
    rows = _rollouts(
        _Policy(), lambda *args, **kwargs: _Env(), run_id="run", kind="specialist_center",
        method="hidden", training_seed=0, split="own_goal", goal=0.0,
        episode_seeds=[5_000_000, 5_000_001], checkpoint=checkpoint,
    )
    assert [row["return"] for row in rows] == [3.0, 3.0]
    assert all(row["episode_length"] == 2 and row["deterministic"] for row in rows)
    summary = _summary(rows)
    assert summary[0]["episodes"] == 2
    assert summary[0]["mean_return"] == 3.0

