from __future__ import annotations

import pytest

from crl_ood.environments.context_splits import (
    build_context_splits,
    carl_feature_key,
    context_values,
)


@pytest.mark.parametrize("feature", ["gravity", "length", "dt"])
def test_context_assignment_is_seeded_and_deterministic(smoke_config, feature):
    split_config = smoke_config["environment"]["splits"]
    first = build_context_splits(feature, split_config, seed=42)
    second = build_context_splits(feature, split_config, seed=42)
    different = build_context_splits(feature, split_config, seed=43)

    assert first == second
    assert context_values(first["train"], feature) != context_values(
        different["train"], feature
    )


@pytest.mark.parametrize("feature", ["gravity", "length", "dt"])
def test_train_and_ood_ranges_are_disjoint(smoke_config, feature):
    splits = build_context_splits(feature, smoke_config["environment"]["splits"], seed=0)
    key = carl_feature_key(feature)
    train = {context[key] for context in splits["train"].values()}
    identity = {context[key] for context in splits["id_test"].values()}
    low = {context[key] for context in splits["ood_low"].values()}
    high = {context[key] for context in splits["ood_high"].values()}

    assert train.isdisjoint(identity)
    assert train.isdisjoint(low)
    assert train.isdisjoint(high)
    assert low.isdisjoint(high)
    assert max(low) < min(train)
    assert min(high) > max(train)
