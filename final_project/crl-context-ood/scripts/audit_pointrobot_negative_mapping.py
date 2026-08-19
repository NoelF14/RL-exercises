from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from crl_ood.pointrobot_encoders.dataset import (
    TRAIN_CONTEXTS,
    load_dataset,
    make_window,
    window_indices,
)
from crl_ood.pointrobot_encoders.training import hard_negative_rewards


# ---------------------------------------------------------------------
# Frozen primary-data specification
# ---------------------------------------------------------------------

EXPECTED_CHECKSUM = (
    "cb826e04b344eb875662b8775b89f9c60bdb9bae895f25a260d25ef422a589fa"
)

HISTORY_LENGTH = 5
FUTURE_HORIZON = 5

OUTDIR = Path("results/pointrobot_p3")


# ---------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------

def find_primary_dataset() -> Path:
    candidates = []

    for checksum_file in Path("results").rglob("dataset.sha256"):
        checksum = checksum_file.read_text(encoding="ascii").strip()

        if checksum == EXPECTED_CHECKSUM:
            candidates.append(checksum_file.parent)

    if len(candidates) != 1:
        print("Matching candidates:", candidates)
        raise RuntimeError(
            "Expected exactly one dataset matching the frozen primary checksum, "
            f"found {len(candidates)}."
        )

    return candidates[0]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def nearest_context_index(context: float) -> int:
    return min(
        range(len(TRAIN_CONTEXTS)),
        key=lambda i: abs(float(TRAIN_CONTEXTS[i]) - float(context)),
    )


def implemented_negative_goal(context: float) -> float:
    source = nearest_context_index(context)
    return float(
        TRAIN_CONTEXTS[(source + 1) % len(TRAIN_CONTEXTS)]
    )


def scalar_context(value) -> float:
    array = np.asarray(value)
    return float(array.reshape(-1)[0])


def label_goal(x: float) -> str:
    return f"{x:+.1f}"


# ---------------------------------------------------------------------
# Direct implementation spot-check
# ---------------------------------------------------------------------

def verify_hard_negative_function(arrays):
    """
    Pick one actual training window for each positive context and call the
    production hard_negative_rewards() implementation directly.
    """
    indices = window_indices(
        arrays,
        "train",
        HISTORY_LENGTH,
        FUTURE_HORIZON,
    )

    representatives = {}

    for index in indices:
        row = make_window(
            arrays,
            index,
            HISTORY_LENGTH,
            FUTURE_HORIZON,
        )

        context = scalar_context(row["context"])
        source = float(
            TRAIN_CONTEXTS[nearest_context_index(context)]
        )

        if source not in representatives:
            representatives[source] = row

        if len(representatives) == len(TRAIN_CONTEXTS):
            break

    assert set(representatives) == set(
        float(x) for x in TRAIN_CONTEXTS
    )

    ordered_rows = [
        representatives[float(context)]
        for context in TRAIN_CONTEXTS
    ]

    batch = {
        "context": torch.stack(
            [
                torch.as_tensor(row["context"])
                for row in ordered_rows
            ]
        ),
        "future_states": torch.stack(
            [
                torch.as_tensor(row["future_states"])
                for row in ordered_rows
            ]
        ),
        "future_actions": torch.stack(
            [
                torch.as_tensor(row["future_actions"])
                for row in ordered_rows
            ]
        ),
        "future_rewards": torch.stack(
            [
                torch.as_tensor(row["future_rewards"])
                for row in ordered_rows
            ]
        ),
    }

    _, provenance = hard_negative_rewards(batch)

    direct = pd.DataFrame(provenance)

    for _, row in direct.iterrows():
        expected = implemented_negative_goal(
            float(row["positive_goal"])
        )

        assert np.isclose(
            float(row["negative_goal"]),
            expected,
        )

        assert bool(row["state_action_preserved"])
        assert bool(row["reward_targets_different"])

    return direct


# ---------------------------------------------------------------------
# Full empirical frequency audit
# ---------------------------------------------------------------------

def enumerate_training_windows(arrays):
    indices = window_indices(
        arrays,
        "train",
        HISTORY_LENGTH,
        FUTURE_HORIZON,
    )

    records = []

    for index in indices:
        row = make_window(
            arrays,
            index,
            HISTORY_LENGTH,
            FUTURE_HORIZON,
        )

        observed_context = scalar_context(row["context"])

        source_index = nearest_context_index(
            observed_context
        )

        positive_goal = float(
            TRAIN_CONTEXTS[source_index]
        )

        # Same deterministic mapping as the production implementation.
        negative_goal = float(
            TRAIN_CONTEXTS[
                (source_index + 1)
                % len(TRAIN_CONTEXTS)
            ]
        )

        assert np.isclose(
            observed_context,
            positive_goal,
        )

        records.append(
            {
                "episode": int(index.episode),
                "timestep": int(index.timestep),
                "positive_goal": positive_goal,
                "negative_goal": negative_goal,
            }
        )

    return pd.DataFrame(records)


def build_matrices(windows):
    goals = [
        float(x)
        for x in TRAIN_CONTEXTS
    ]

    counts = pd.crosstab(
        windows["positive_goal"],
        windows["negative_goal"],
    ).reindex(
        index=goals,
        columns=goals,
        fill_value=0,
    )

    counts.index.name = "positive_goal"
    counts.columns.name = "negative_goal"

    frequencies = counts.div(
        counts.sum(axis=1),
        axis=0,
    )

    # Every positive goal must map to exactly one negative.
    nonzero_per_row = (
        counts.to_numpy() > 0
    ).sum(axis=1)

    assert np.all(nonzero_per_row == 1)

    # Confirm the only occupied cell in each row matches production mapping.
    for positive in goals:
        occupied = [
            negative
            for negative in goals
            if counts.loc[positive, negative] > 0
        ]

        assert len(occupied) == 1

        expected = implemented_negative_goal(
            positive
        )

        assert np.isclose(
            occupied[0],
            expected,
        )

    return counts, frequencies


def build_marginals(windows):
    positive = (
        windows.groupby("positive_goal")
        .size()
        .rename("positive_window_count")
    )

    negative = (
        windows.groupby("negative_goal")
        .size()
        .rename("negative_window_count")
    )

    goals = [
        float(x)
        for x in TRAIN_CONTEXTS
    ]

    marginal = pd.DataFrame(
        index=goals
    )

    marginal.index.name = "goal"

    marginal = marginal.join(
        positive,
        how="left",
    )

    marginal = marginal.join(
        negative,
        how="left",
    )

    marginal = marginal.fillna(0).astype(int)

    total = len(windows)

    marginal["positive_frequency"] = (
        marginal["positive_window_count"]
        / total
    )

    marginal["negative_frequency"] = (
        marginal["negative_window_count"]
        / total
    )

    return marginal


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------

def plot_heatmap(counts, frequencies):
    goals = [
        float(x)
        for x in TRAIN_CONTEXTS
    ]

    matrix = frequencies.to_numpy()

    fig, ax = plt.subplots(
        figsize=(4.4, 3.7)
    )

    image = ax.imshow(
        matrix,
        vmin=0.0,
        vmax=1.0,
        cmap="Blues",
        aspect="equal",
    )

    ax.set_xticks(
        np.arange(len(goals))
    )
    ax.set_yticks(
        np.arange(len(goals))
    )

    ax.set_xticklabels(
        [label_goal(x) for x in goals]
    )
    ax.set_yticklabels(
        [label_goal(x) for x in goals]
    )

    ax.set_xlabel(
        r"Negative goal $\varphi^-$"
    )
    ax.set_ylabel(
        r"Positive goal $\varphi^+$"
    )

    for i in range(len(goals)):
        for j in range(len(goals)):
            frequency = float(
                matrix[i, j]
            )

            count = int(
                counts.iloc[i, j]
            )

            if count == 0:
                text = "0"
            else:
                text = (
                    f"{frequency:.2f}\n"
                    f"n={count:,}"
                )

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=7.5,
                color=(
                    "white"
                    if frequency > 0.5
                    else "black"
                ),
            )

    cbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )
    cbar.set_label(
        "Row-normalized frequency"
    )

    fig.tight_layout()

    fig.savefig(
        OUTDIR
        / "negative_mapping_heatmap.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        OUTDIR
        / "negative_mapping_heatmap.png",
        dpi=400,
        bbox_inches="tight",
    )

    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_dir = find_primary_dataset()

    print(
        "Primary dataset:",
        dataset_dir,
    )

    arrays, metadata = load_dataset(
        dataset_dir
    )

    assert (
        metadata["dataset_checksum"]
        == EXPECTED_CHECKSUM
    )

    print(
        "Dataset checksum:",
        metadata["dataset_checksum"],
    )

    print(
        "TRAIN_CONTEXTS:",
        TRAIN_CONTEXTS,
    )

    # Direct call into the production negative-generation function.
    direct = verify_hard_negative_function(
        arrays
    )

    direct.to_csv(
        OUTDIR
        / "negative_mapping_direct_provenance.csv",
        index=False,
    )

    print(
        "\n=== Direct production-function verification ==="
    )
    print(
        direct[
            [
                "positive_goal",
                "negative_goal",
                "state_action_preserved",
                "reward_targets_different",
            ]
        ].to_string(index=False)
    )

    # Enumerate every actual encoder-training window.
    windows = enumerate_training_windows(
        arrays
    )

    counts, frequencies = build_matrices(
        windows
    )

    marginals = build_marginals(
        windows
    )

    counts.to_csv(
        OUTDIR
        / "negative_mapping_counts.csv"
    )

    frequencies.to_csv(
        OUTDIR
        / "negative_mapping_frequencies.csv"
    )

    marginals.to_csv(
        OUTDIR
        / "negative_mapping_marginals.csv"
    )

    plot_heatmap(
        counts,
        frequencies,
    )

    print(
        "\n=== Training windows ==="
    )
    print(
        f"Total: {len(windows):,}"
    )

    print(
        "\n=== Positive -> negative counts ==="
    )
    print(
        counts.to_string()
    )

    print(
        "\n=== Row-normalized frequencies ==="
    )
    print(
        frequencies.to_string(
            float_format=lambda x: f"{x:.3f}"
        )
    )

    print(
        "\n=== Marginal frequencies ==="
    )
    print(
        marginals.to_string(
            float_format=lambda x: f"{x:.6f}"
        )
    )

    positive_counts = (
        marginals[
            "positive_window_count"
        ].to_numpy()
    )

    imbalance_ratio = (
        positive_counts.max()
        / positive_counts.min()
    )

    print(
        "\nPositive-context max/min count ratio:",
        f"{imbalance_ratio:.6f}",
    )

    print(
        "\nExpected directed mapping:"
    )

    for positive in TRAIN_CONTEXTS:
        negative = implemented_negative_goal(
            float(positive)
        )

        signed_shift = (
            negative
            - float(positive)
        )

        print(
            f"{positive:+.1f}"
            f" -> {negative:+.1f}"
            f"    shift={signed_shift:+.1f}"
        )

    print(
        "\nWrote:"
    )

    for filename in [
        "negative_mapping_direct_provenance.csv",
        "negative_mapping_counts.csv",
        "negative_mapping_frequencies.csv",
        "negative_mapping_marginals.csv",
        "negative_mapping_heatmap.pdf",
        "negative_mapping_heatmap.png",
    ]:
        print(
            OUTDIR / filename
        )


if __name__ == "__main__":
    main()
