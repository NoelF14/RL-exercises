from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


PROBE_INPUT = Path(
    "results/pointrobot_representation/representation_probe_by_angle.csv"
)
STATE_INPUT = Path(
    "results/pointrobot_representation/representation_state_only_probe.csv"
)
OUTDIR = Path("figures")

EXPECTED_METHODS = ["vae", "contrastive"]
EXPECTED_ANGLES = np.array(
    [-1.0, -0.8, -0.6, -0.45, -0.3, -0.15, 0.0,
      0.15, 0.3, 0.45, 0.6, 0.8, 1.0]
)

DISPLAY_NAME = {
    "vae": "VAE",
    "contrastive": "Contrastive",
    "state_only": "State only",
}

# Match the return-by-angle panel exactly.
STYLE = {
    "state_only": {
        "color": "0.45",
        "marker": "s",
        "linestyle": "--",
        "linewidth": 1.3,
    },
    "vae": {
        "color": "tab:blue",
        "marker": "^",
        "linestyle": "-",
        "linewidth": 1.5,
    },
    "contrastive": {
        "color": "tab:orange",
        "marker": "D",
        "linestyle": "-",
        "linewidth": 1.5,
    },
}

PLOT_ORDER = ["state_only", "vae", "contrastive"]


def expected_split_for_angle(angle):
    mapping = {
        -1.00: "ood_left",
        -0.80: "ood_left",
        -0.60: "train",
        -0.45: "id",
        -0.30: "train",
        -0.15: "id",
         0.00: "train",
         0.15: "id",
         0.30: "train",
         0.45: "id",
         0.60: "train",
         0.80: "ood_right",
         1.00: "ood_right",
    }
    return mapping[float(angle)]


def load_probe():
    df = pd.read_csv(PROBE_INPUT)

    required = {
        "split",
        "goal_angle",
        "circular_angle_mae",
        "fit_split",
        "method",
        "encoder_seed",
    }
    missing = required - set(df.columns)
    assert not missing, f"Probe file missing columns: {sorted(missing)}"

    df = df[df["method"].isin(EXPECTED_METHODS)].copy()

    assert set(df["method"]) == set(EXPECTED_METHODS)
    assert set(df["fit_split"]) == {"train"}

    # Exactly one angle-level result for each method / seed / angle.
    assert not df.duplicated(
        ["method", "encoder_seed", "goal_angle"]
    ).any()

    for method in EXPECTED_METHODS:
        sub = df[df["method"] == method]

        seeds = sorted(sub["encoder_seed"].astype(int).unique())
        assert seeds == [0, 1, 2, 3, 4], (method, seeds)

        angles = np.sort(sub["goal_angle"].astype(float).unique())
        assert np.allclose(angles, EXPECTED_ANGLES), (method, angles)

    for angle in EXPECTED_ANGLES:
        actual = set(
            df.loc[
                np.isclose(df["goal_angle"].astype(float), angle),
                "split",
            ]
        )
        expected = {expected_split_for_angle(angle)}
        assert actual == expected, (angle, actual, expected)

    return df


def aggregate_probe(df):
    summary = (
        df.groupby(["method", "split", "goal_angle"], as_index=False)
        .agg(
            mean_mae=("circular_angle_mae", "mean"),
            std_mae=("circular_angle_mae", lambda x: x.std(ddof=1)),
            seed_count=("encoder_seed", "nunique"),
        )
        .sort_values(["method", "goal_angle"])
    )

    assert (summary["seed_count"] == 5).all()
    return summary


def load_state_only():
    df = pd.read_csv(
        STATE_INPUT,
        usecols=[
            "method",
            "encoder_seed",
            "split",
            "goal_angle",
            "probe_fit_split",
            "features",
            "contains_history",
            "angle_circular_angle_mae",
        ],
    )

    assert set(df["probe_fit_split"]) == {"train"}
    assert set(df["features"]) == {"current_state_only"}

    # CSV may encode boolean values as Python bools or strings.
    contains_history = (
        df["contains_history"]
        .astype(str)
        .str.lower()
        .unique()
        .tolist()
    )
    assert set(contains_history) == {"false"}

    # The state-only baseline is repeated under method/encoder bookkeeping.
    # It is NOT five independent representation estimates.
    #
    # Verify that every duplicate gives exactly the same angle-level value,
    # then retain one unique value per goal angle.
    nunique = (
        df.groupby("goal_angle")["angle_circular_angle_mae"]
        .nunique()
    )
    assert (nunique == 1).all(), nunique

    state = (
        df[
            [
                "split",
                "goal_angle",
                "angle_circular_angle_mae",
            ]
        ]
        .drop_duplicates()
        .sort_values("goal_angle")
        .reset_index(drop=True)
    )

    # After removing duplicated bookkeeping, there must be exactly 13 angles.
    assert len(state) == len(EXPECTED_ANGLES)

    angles = np.sort(state["goal_angle"].astype(float).unique())
    assert np.allclose(angles, EXPECTED_ANGLES), angles

    for angle in EXPECTED_ANGLES:
        actual = set(
            state.loc[
                np.isclose(state["goal_angle"].astype(float), angle),
                "split",
            ]
        )
        expected = {expected_split_for_angle(angle)}
        assert actual == expected, (angle, actual, expected)

    state = state.rename(
        columns={"angle_circular_angle_mae": "mean_mae"}
    )

    return state


def decorate_context_regions(ax):
    # Same OOD shading used in the return panel.
    ax.axvspan(-1.05, -0.60, color="0.94", zorder=0)
    ax.axvspan(0.60, 1.05, color="0.94", zorder=0)

    # Boundaries of training-context range.
    ax.axvline(-0.60, color="0.7", linewidth=0.8, linestyle=":")
    ax.axvline(0.60, color="0.7", linewidth=0.8, linestyle=":")

    ax.grid(axis="y", linewidth=0.5, alpha=0.25)
    ax.set_xlim(-1.05, 1.05)


def draw_learned(ax, summary, method):
    d = (
        summary[summary["method"] == method]
        .sort_values("goal_angle")
    )

    st = STYLE[method]

    ax.errorbar(
        d["goal_angle"],
        d["mean_mae"],
        yerr=d["std_mae"],
        color=st["color"],
        linestyle=st["linestyle"],
        linewidth=st["linewidth"],
        marker=st["marker"],
        markersize=4.0,
        markeredgewidth=0.8,
        capsize=2.0,
        elinewidth=0.8,
        alpha=0.95,
        zorder=3,
    )


def draw_state_only(ax, state):
    st = STYLE["state_only"]

    # No error bars: repeated rows are bookkeeping duplicates,
    # not independent state-only estimates.
    ax.plot(
        state["goal_angle"],
        state["mean_mae"],
        color=st["color"],
        linestyle=st["linestyle"],
        linewidth=st["linewidth"],
        marker=st["marker"],
        markersize=4.0,
        markeredgewidth=0.8,
        alpha=0.95,
        zorder=2,
    )


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    probe = load_probe()
    summary = aggregate_probe(probe)
    state = load_state_only()

    # Save exact values used by the report figure.
    summary.to_csv(
        OUTDIR / "pointrobot_probe_by_angle_summary.csv",
        index=False,
    )
    state.to_csv(
        OUTDIR / "pointrobot_state_only_by_angle_summary.csv",
        index=False,
    )

    # Match width/overall scale of the return panel.
    fig, ax = plt.subplots(figsize=(5.2, 3.15))

    decorate_context_regions(ax)

    draw_state_only(ax, state)
    draw_learned(ax, summary, "vae")
    draw_learned(ax, summary, "contrastive")

    ax.set_ylabel("Linear probe MAE (rad)")
    ax.set_xlabel(r"Goal angle $\varphi$ (rad)")

    # State-only reaches exactly 1 rad at +/-1.
    # Keep the full baseline visible rather than clipping it.
    ax.set_ylim(0.0, 1.05)

    # Same sparse x tick labels as the return panel.
    major_ticks = [-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0]
    ax.set_xticks(major_ticks)

    # Region labels match return panel placement.
    for x, label in [
        (-0.82, "OOD-L"),
        (0.00, "Train / ID"),
        (0.82, "OOD-R"),
    ]:
        ax.text(
            x,
            1.015,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=7.5,
            clip_on=False,
        )

    handles = [
        Line2D(
            [0],
            [0],
            color=STYLE[m]["color"],
            linestyle=STYLE[m]["linestyle"],
            marker=STYLE[m]["marker"],
            linewidth=STYLE[m]["linewidth"],
            markersize=4,
            label=DISPLAY_NAME[m],
        )
        for m in PLOT_ORDER
    ]

    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=3,
        frameon=False,
        fontsize=8,
        columnspacing=1.1,
        handlelength=2.0,
    )

    fig.savefig(
        OUTDIR / "pointrobot_probe_by_angle.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        OUTDIR / "pointrobot_probe_by_angle.png",
        dpi=400,
        bbox_inches="tight",
    )

    print("=== Learned representations: mean +/- SD across 5 encoder seeds ===")
    print(summary.to_string(index=False))

    print("\n=== State-only baseline: unique angle-level values ===")
    print(state.to_string(index=False))

    print("\nWrote:")
    print(OUTDIR / "pointrobot_probe_by_angle.pdf")
    print(OUTDIR / "pointrobot_probe_by_angle.png")
    print(OUTDIR / "pointrobot_probe_by_angle_summary.csv")
    print(OUTDIR / "pointrobot_state_only_by_angle_summary.csv")


if __name__ == "__main__":
    main()
