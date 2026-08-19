from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crl_ood.pointrobot_encoders.dataset import (
    TRAIN_CONTEXTS,
    load_dataset,
    make_window,
    pointrobot_reward,
    window_indices,
)


EXPECTED_CHECKSUM = (
    "cb826e04b344eb875662b8775b89f9c60bdb9bae895f25a260d25ef422a589fa"
)

HISTORY_LENGTH = 5
FUTURE_HORIZON = 5

OUTDIR = Path("results/pointrobot_p3")


def find_primary_dataset() -> Path:
    candidates = []

    for checksum_file in Path("results").rglob("dataset.sha256"):
        checksum = checksum_file.read_text(
            encoding="ascii"
        ).strip()

        if checksum == EXPECTED_CHECKSUM:
            candidates.append(
                checksum_file.parent
            )

    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one primary dataset; "
            f"found {candidates}"
        )

    return candidates[0]


def scalar(value) -> float:
    return float(
        np.asarray(value).reshape(-1)[0]
    )


def source_index(context: float) -> int:
    return min(
        range(len(TRAIN_CONTEXTS)),
        key=lambda i: abs(
            float(TRAIN_CONTEXTS[i])
            - float(context)
        ),
    )


def negative_goal_for(context: float) -> float:
    source = source_index(context)

    return float(
        TRAIN_CONTEXTS[
            (source + 1)
            % len(TRAIN_CONTEXTS)
        ]
    )


def pair_label(
    positive: float,
    negative: float,
) -> str:
    return (
        f"{positive:+.1f}"
        "\u2192"
        f"{negative:+.1f}"
    )


def enumerate_window_metrics(arrays):
    indices = window_indices(
        arrays,
        "train",
        HISTORY_LENGTH,
        FUTURE_HORIZON,
    )

    rows = []

    context_key_sets = {
        float(c): set()
        for c in TRAIN_CONTEXTS
    }

    for index in indices:
        window = make_window(
            arrays,
            index,
            HISTORY_LENGTH,
            FUTURE_HORIZON,
        )

        observed_context = scalar(
            window["context"]
        )

        src = source_index(
            observed_context
        )

        positive_goal = float(
            TRAIN_CONTEXTS[src]
        )

        negative_goal = float(
            TRAIN_CONTEXTS[
                (src + 1)
                % len(TRAIN_CONTEXTS)
            ]
        )

        assert np.isclose(
            observed_context,
            positive_goal,
        )

        future_states = np.asarray(
            window["future_states"]
        )

        future_actions = np.asarray(
            window["future_actions"]
        )

        positive_rewards = np.asarray(
            window["future_rewards"],
            dtype=np.float64,
        ).reshape(-1)

        negative_rewards = np.asarray(
            pointrobot_reward(
                future_states,
                future_actions,
                negative_goal,
            ),
            dtype=np.float64,
        ).reshape(-1)

        assert (
            len(positive_rewards)
            == FUTURE_HORIZON
        )

        assert (
            negative_rewards.shape
            == positive_rewards.shape
        )

        delta = (
            negative_rewards
            - positive_rewards
        )

        trajectory_seed = int(
            arrays["trajectory_seeds"][
                index.episode
            ]
        )

        key = (
            trajectory_seed,
            int(index.timestep),
        )

        context_key_sets[
            positive_goal
        ].add(key)

        rows.append(
            {
                "episode": int(
                    index.episode
                ),
                "trajectory_seed": (
                    trajectory_seed
                ),
                "timestep": int(
                    index.timestep
                ),
                "positive_goal": (
                    positive_goal
                ),
                "negative_goal": (
                    negative_goal
                ),
                "signed_goal_shift": (
                    negative_goal
                    - positive_goal
                ),
                "angular_separation": abs(
                    negative_goal
                    - positive_goal
                ),
                "goal_vector_distance": (
                    2.0
                    * np.sin(
                        abs(
                            negative_goal
                            - positive_goal
                        )
                        / 2.0
                    )
                ),
                "reward_delta_mean": (
                    float(
                        np.mean(delta)
                    )
                ),
                "reward_delta_mean_abs": (
                    float(
                        np.mean(
                            np.abs(delta)
                        )
                    )
                ),
                "reward_delta_rmse": (
                    float(
                        np.sqrt(
                            np.mean(
                                delta ** 2
                            )
                        )
                    )
                ),
                "reward_delta_l2": (
                    float(
                        np.linalg.norm(
                            delta
                        )
                    )
                ),
                "reward_delta_max_abs": (
                    float(
                        np.max(
                            np.abs(delta)
                        )
                    )
                ),
            }
        )

    # All five contexts should be evaluated on exactly
    # the same trajectory-seed/timestep keys.
    reference = context_key_sets[
        float(TRAIN_CONTEXTS[0])
    ]

    for context, keys in (
        context_key_sets.items()
    ):
        assert keys == reference, (
            context,
            len(keys),
            len(reference),
        )

    return pd.DataFrame(rows)


def aggregate_by_trajectory(windows):
    # One diagnostic value per independently generated
    # behavior trajectory rather than per overlapping window.
    return (
        windows.groupby(
            [
                "positive_goal",
                "negative_goal",
                "trajectory_seed",
            ],
            as_index=False,
        )
        .agg(
            mean_window_rmse=(
                "reward_delta_rmse",
                "mean",
            ),
            mean_abs_reward_delta=(
                "reward_delta_mean_abs",
                "mean",
            ),
            mean_signed_reward_delta=(
                "reward_delta_mean",
                "mean",
            ),
            mean_max_abs_reward_delta=(
                "reward_delta_max_abs",
                "mean",
            ),
            window_count=(
                "timestep",
                "count",
            ),
        )
    )


def build_summary(trajectories):
    rows = []

    for positive in TRAIN_CONTEXTS:
        positive = float(positive)
        negative = negative_goal_for(
            positive
        )

        values = trajectories.loc[
            np.isclose(
                trajectories[
                    "positive_goal"
                ],
                positive,
            ),
            "mean_window_rmse",
        ].to_numpy()

        signed = trajectories.loc[
            np.isclose(
                trajectories[
                    "positive_goal"
                ],
                positive,
            ),
            "mean_signed_reward_delta",
        ].to_numpy()

        mean_abs = trajectories.loc[
            np.isclose(
                trajectories[
                    "positive_goal"
                ],
                positive,
            ),
            "mean_abs_reward_delta",
        ].to_numpy()

        assert len(values) == 320

        separation = abs(
            negative
            - positive
        )

        rows.append(
            {
                "positive_goal": positive,
                "negative_goal": negative,
                "pair": pair_label(
                    positive,
                    negative,
                ),
                "signed_goal_shift": (
                    negative
                    - positive
                ),
                "angular_separation": (
                    separation
                ),
                "goal_vector_distance": (
                    2.0
                    * np.sin(
                        separation
                        / 2.0
                    )
                ),
                "trajectory_count": (
                    len(values)
                ),
                "reward_rmse_mean": (
                    values.mean()
                ),
                "reward_rmse_sd": (
                    values.std(ddof=1)
                ),
                "reward_rmse_median": (
                    np.median(values)
                ),
                "reward_rmse_q25": (
                    np.quantile(
                        values,
                        0.25,
                    )
                ),
                "reward_rmse_q75": (
                    np.quantile(
                        values,
                        0.75,
                    )
                ),
                "mean_abs_delta": (
                    mean_abs.mean()
                ),
                "mean_signed_delta": (
                    signed.mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def plot_signal(
    trajectories,
    summary,
):
    pairs = summary[
        "pair"
    ].tolist()

    data = []

    for positive in TRAIN_CONTEXTS:
        positive = float(positive)

        values = trajectories.loc[
            np.isclose(
                trajectories[
                    "positive_goal"
                ],
                positive,
            ),
            "mean_window_rmse",
        ].to_numpy()

        data.append(values)

    fig, ax = plt.subplots(
        figsize=(5.2, 3.2)
    )

    ax.boxplot(
        data,
        tick_labels=pairs,
        showfliers=False,
        widths=0.55,
    )

    means = summary[
        "reward_rmse_mean"
    ].to_numpy()

    ax.scatter(
        np.arange(
            1,
            len(means) + 1,
        ),
        means,
        marker="D",
        s=22,
        zorder=3,
        label="Mean",
    )

    ax.set_ylabel(
        "Mean 5-step reward-difference RMSE"
    )

    ax.set_xlabel(
        r"Positive $\rightarrow$ negative goal"
    )

    ax.grid(
        axis="y",
        linewidth=0.5,
        alpha=0.25,
    )

    ax.legend(
        frameon=False,
        fontsize=8,
    )

    fig.tight_layout()

    fig.savefig(
        OUTDIR
        / "reward_relabel_signal.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        OUTDIR
        / "reward_relabel_signal.png",
        dpi=400,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_dir = (
        find_primary_dataset()
    )

    print(
        "Primary dataset:",
        dataset_dir,
    )

    arrays, metadata = (
        load_dataset(
            dataset_dir
        )
    )

    assert (
        metadata[
            "dataset_checksum"
        ]
        == EXPECTED_CHECKSUM
    )

    windows = (
        enumerate_window_metrics(
            arrays
        )
    )

    trajectories = (
        aggregate_by_trajectory(
            windows
        )
    )

    summary = build_summary(
        trajectories
    )

    assert (
        windows.groupby(
            "positive_goal"
        ).size()
        == 14720
    ).all()

    assert (
        trajectories.groupby(
            "positive_goal"
        ).size()
        == 320
    ).all()

    assert (
        trajectories[
            "window_count"
        ]
        == 46
    ).all()

    windows.to_csv(
        OUTDIR
        / "reward_relabel_window_metrics.csv",
        index=False,
    )

    trajectories.to_csv(
        OUTDIR
        / "reward_relabel_trajectory_metrics.csv",
        index=False,
    )

    summary.to_csv(
        OUTDIR
        / "reward_relabel_pair_summary.csv",
        index=False,
    )

    plot_signal(
        trajectories,
        summary,
    )

    print(
        "\n=== Reward-relabelling "
        "signal by directed pair ==="
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )

    neighbor = summary[
        np.isclose(
            summary[
                "angular_separation"
            ],
            0.3,
        )
    ]

    wrap = summary[
        np.isclose(
            summary[
                "positive_goal"
            ],
            0.6,
        )
    ].iloc[0]

    neighbor_mean = (
        neighbor[
            "reward_rmse_mean"
        ].mean()
    )

    ratio = (
        wrap[
            "reward_rmse_mean"
        ]
        / neighbor_mean
    )

    print(
        "\nMean RMSE across "
        "four +0.3 neighbor edges:",
        f"{neighbor_mean:.6f}",
    )

    print(
        "RMSE for +0.6 -> -0.6:",
        f"{wrap['reward_rmse_mean']:.6f}",
    )

    print(
        "Wrap / neighbor signal ratio:",
        f"{ratio:.3f}x",
    )

    print(
        "\nReminder: larger reward RMSE "
        "means a stronger raw reward-only "
        "difference, not necessarily easier "
        "neural optimization."
    )

    print(
        "\nWrote:"
    )

    for name in [
        "reward_relabel_window_metrics.csv",
        "reward_relabel_trajectory_metrics.csv",
        "reward_relabel_pair_summary.csv",
        "reward_relabel_signal.pdf",
        "reward_relabel_signal.png",
    ]:
        print(
            OUTDIR / name
        )


if __name__ == "__main__":
    main()
