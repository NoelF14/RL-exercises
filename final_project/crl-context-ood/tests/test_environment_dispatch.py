import pytest

from crl_ood.environments.context_splits import (
    build_context_splits,
    carl_feature_key,
)
from crl_ood.environments import factory


def _explicit_splits():
    return {
        "train": {"values": [0.8, 1.0, 1.2]},
        "id_test": {"values": [0.9, 1.1]},
        "ood_low": {"values": [0.6]},
        "ood_high": {"values": [1.4]},
    }


def test_feature_keys_are_environment_specific():
    assert carl_feature_key("length", "pendulum") == "l"
    assert carl_feature_key("length", "cartpole") == "length"
    assert carl_feature_key("gravity", "pendulum") == "g"
    assert carl_feature_key("dt", "pendulum") == "dt"

    with pytest.raises(ValueError):
        carl_feature_key("gravity", "cartpole")


def test_context_splits_use_correct_environment_key():
    pendulum = build_context_splits(
        "length",
        _explicit_splits(),
        seed=7,
        environment="pendulum",
    )

    cartpole = build_context_splits(
        "length",
        _explicit_splits(),
        seed=7,
        environment="cartpole",
    )

    assert all(
        set(context) == {"l"}
        for split in pendulum.values()
        for context in split.values()
    )

    assert all(
        set(context) == {"length"}
        for split in cartpole.values()
        for context in split.values()
    )


def test_make_env_dispatches_by_environment(monkeypatch):
    calls = []

    def fake_pendulum(*args, **kwargs):
        calls.append("pendulum")
        return "pendulum-env"

    def fake_cartpole(*args, **kwargs):
        calls.append("cartpole")
        return "cartpole-env"

    monkeypatch.setattr(
        factory,
        "make_pendulum_env",
        fake_pendulum,
    )
    monkeypatch.setattr(
        factory,
        "make_cartpole_env",
        fake_cartpole,
    )

    common = dict(
        feature="length",
        mode="hidden",
        seed=0,
        context_normalization=(1.0, 0.2),
    )

    assert factory.make_env(
        "pendulum",
        contexts={0: {"l": 1.0}},
        **common,
    ) == "pendulum-env"

    assert factory.make_env(
        "cartpole",
        contexts={0: {"length": 1.0}},
        **common,
    ) == "cartpole-env"

    assert calls == [
        "pendulum",
        "cartpole",
    ]
