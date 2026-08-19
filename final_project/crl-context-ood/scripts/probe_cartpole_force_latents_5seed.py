from pathlib import Path

import numpy as np
import pandas as pd
import torch

from crl_ood.pointrobot_encoders.dataset import (
    load_dataset,
    make_window,
    window_indices,
)
from crl_ood.pointrobot_encoders.training import (
    load_frozen_checkpoint,
)


SEEDS = [0, 1, 2, 3, 4]

DATASET = Path(
    "results/cartpole_force_encoders/"
    "dataset/full_frozen"
)

ENCODERS = Path(
    "results/cartpole_force_encoders/"
    "encoders"
)

ANALYSIS = Path(
    "results/cartpole_force_encoders/"
    "analysis"
)

CHECKSUM = (
    "be2942887fd919057efb89525fcc8809c0d7dfb5ae6985020520cbabfd578ebf"
)

H = 5
K = 5
RIDGE = 1e-3
BATCH = 512


arrays, metadata = load_dataset(
    DATASET
)

assert (
    metadata["dataset_checksum"]
    == CHECKSUM
)


def get_samples(assignment):
    histories = []
    states = []
    force = []

    for index in window_indices(
        arrays,
        assignment,
        H,
        K,
    ):
        row = make_window(
            arrays,
            index,
            H,
            K,
        )

        if int(row["length"]) != H:
            continue

        histories.append(
            row["history"]
        )
        states.append(
            row["current_state"]
        )
        force.append(
            float(row["context"])
        )

    return (
        np.asarray(
            histories,
            dtype=np.float32,
        ),
        np.asarray(
            states,
            dtype=np.float64,
        ),
        np.asarray(
            force,
            dtype=np.float64,
        ),
    )


train_h, train_s, train_y = (
    get_samples("train")
)

val_h, val_s, val_y = (
    get_samples("validation")
)


def fit_ridge(X, y):
    mean = X.mean(axis=0)
    std = X.std(axis=0)

    std[std < 1e-10] = 1.0

    Z = (
        X - mean
    ) / std

    y_mean = y.mean()

    beta = np.linalg.solve(
        Z.T @ Z
        + RIDGE * np.eye(
            Z.shape[1]
        ),
        Z.T @ (
            y - y_mean
        ),
    )

    return (
        mean,
        std,
        y_mean,
        beta,
    )


def predict(model, X):
    mean, std, y_mean, beta = model

    return (
        (X - mean)
        / std
    ) @ beta + y_mean


def metrics(y, pred):
    error = pred - y

    return {
        "mae":
            float(
                np.mean(
                    np.abs(error)
                )
            ),
        "rmse":
            float(
                np.sqrt(
                    np.mean(
                        error ** 2
                    )
                )
            ),
        "r2":
            float(
                1
                - np.sum(
                    error ** 2
                )
                / np.sum(
                    (
                        y
                        - y.mean()
                    ) ** 2
                )
            ),
    }


def encode(model, history):
    chunks = []

    model.eval()

    with torch.no_grad():
        for start in range(
            0,
            len(history),
            BATCH,
        ):
            x = torch.from_numpy(
                history[
                    start:start+BATCH
                ]
            )

            n = len(x)

            lengths = torch.full(
                (n,),
                H,
                dtype=torch.long,
            )

            mask = torch.ones(
                (n, H),
                dtype=torch.bool,
            )

            z = model.encode(
                x,
                lengths,
                mask,
                deterministic=True,
            )

            chunks.append(
                z.cpu().numpy()
            )

    return np.concatenate(
        chunks,
        axis=0,
    ).astype(np.float64)


rows = []


# Fixed baselines
state_probe = fit_ridge(
    train_s,
    train_y,
)

rows.append(
    {
        "method": "state_only",
        "seed": -1,
        **metrics(
            val_y,
            predict(
                state_probe,
                val_s,
            ),
        ),
    }
)


raw_train = train_h.reshape(
    len(train_h),
    -1,
).astype(np.float64)

raw_val = val_h.reshape(
    len(val_h),
    -1,
).astype(np.float64)

raw_probe = fit_ridge(
    raw_train,
    train_y,
)

rows.append(
    {
        "method": "raw_history_H5",
        "seed": -1,
        **metrics(
            val_y,
            predict(
                raw_probe,
                raw_val,
            ),
        ),
    }
)


for method in [
    "vae",
    "contrastive",
]:
    for seed in SEEDS:

        checkpoint = (
            ENCODERS
            / method
            / f"seed_{seed}"
            / "best.pt"
        )

        model, _ = (
            load_frozen_checkpoint(
                checkpoint,
                expected_method=method,
                expected_dataset_checksum=CHECKSUM,
            )
        )

        z_train = encode(
            model,
            train_h,
        )

        z_val = encode(
            model,
            val_h,
        )

        assert z_train.shape[1] == 8

        probe = fit_ridge(
            z_train,
            train_y,
        )

        result = metrics(
            val_y,
            predict(
                probe,
                z_val,
            ),
        )

        rows.append(
            {
                "method": method,
                "seed": seed,
                **result,
            }
        )

        print(
            f"{method:12s} seed={seed} "
            f"MAE={result['mae']:.5f} "
            f"R2={result['r2']:.5f}"
        )


results = pd.DataFrame(
    rows
)

encoder = results[
    results["method"].isin(
        ["vae", "contrastive"]
    )
]

summary = (
    encoder
    .groupby(
        "method",
        as_index=False,
    )
    .agg(
        mae_mean=("mae", "mean"),
        mae_sd=("mae", "std"),
        rmse_mean=("rmse", "mean"),
        rmse_sd=("rmse", "std"),
        r2_mean=("r2", "mean"),
        r2_sd=("r2", "std"),
        r2_min=("r2", "min"),
        r2_max=("r2", "max"),
    )
)


print()
print("=" * 84)
print("FIVE-SEED LATENT FORCE PROBE")
print("=" * 84)

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}",
    )
)

print()
print("=" * 84)
print("FIVE-SEED ENCODER SUMMARY")
print("=" * 84)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}",
    )
)


results.to_csv(
    ANALYSIS
    / "cartpole_force_latent_probe_5seed_by_seed.csv",
    index=False,
)

summary.to_csv(
    ANALYSIS
    / "cartpole_force_latent_probe_5seed_summary.csv",
    index=False,
)
