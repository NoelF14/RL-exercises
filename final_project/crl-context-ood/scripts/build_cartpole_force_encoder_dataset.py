from pathlib import Path

from crl_ood.cartpole_encoders.dataset import (
    collect_dataset,
)
from crl_ood.pointrobot_encoders.dataset import (
    save_dataset,
)


PHASE0_CONFIG = Path(
    "configs/cartpole_force_phase0.yaml"
)

OUTPUT = Path(
    "results/cartpole_force_encoders/"
    "dataset/full_frozen"
)


arrays, metadata = collect_dataset(
    PHASE0_CONFIG,
    horizon=20,
    trajectories_per_context=200,
    candidate_seed_offset=30_000,
    validation_fraction=0.20,
    split_seed=31,
    max_candidates=10_000,
)

save_dataset(
    OUTPUT,
    arrays,
    metadata,
)

print("dataset:", OUTPUT)
print(
    "checksum:",
    metadata["dataset_checksum"],
)
print(
    "contexts:",
    metadata["contexts"],
)
print(
    "episodes:",
    metadata["episode_count"],
)
print(
    "transitions:",
    metadata["transition_count"],
)
print(
    "accepted/tested:",
    metadata["behavior_policy"]["accepted"],
    "/",
    metadata["behavior_policy"]["candidates_tested"],
)
print(
    "acceptance:",
    metadata["behavior_policy"]["acceptance_fraction"],
)
print(
    "state shape:",
    arrays["states"].shape,
)
print(
    "action shape:",
    arrays["actions"].shape,
)
print(
    "normalization dim:",
    len(arrays["normalization_mean"]),
)
