from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

CONTROL_INPUT = Path(
    "results/pointrobot_primary/primary_context_results.csv"
)

PROBE_INPUT = Path(
    "results/pointrobot_representation/representation_probe_by_angle.csv"
)

STATE_INPUT = Path(
    "results/pointrobot_representation/representation_state_only_probe.csv"
)

OUTDIR = Path("figures")


# ---------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------

EXPECTED_ANGLES = np.array(
    [-1.0, -0.8, -0.6, -0.45, -0.3, -0.15, 0.0,
      0.15, 0.3, 0.45, 0.6, 0.8, 1.0]
)

CONTROL_METHODS = [
    "oracle",
    "no_context",
    "vae",
    "contrastive",
]

LEARNED_METHODS = [
    "vae",
    "contrastive",
]

DISPLAY_NAME = {
    "oracle": "Oracle",
    "no_context": "No context",
    "state_only": "State only",
    "vae": "VAE",
    "contrastive": "Contrastive",
}

STYLE = {
    "oracle": {
        "color": "black",
        "marker": "o",
        "linestyle": "-",
        "linewidth": 1.25,
    },
    "no_context": {
        "color": "0.45",
        "marker": "s",
        "linestyle": "--",
        "linewidth": 1.15,
    },
    "state_only": {
        "color": "0.45",
        "marker": "s",
        "linestyle": "--",
        "linewidth": 1.15,
    },
    "vae": {
        "color": "tab:blue",
        "marker": "^",
        "linestyle": "-",
        "linewidth": 1.35,
    },
    "contrastive": {
        "color": "tab:orange",
        "marker": "D",
        "linestyle": "-",
        "linewidth": 1.35,
    },
}


def expected_split(angle):
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


# ---------------------------------------------------------------------
# Control data
# ---------------------------------------------------------------------

def load_control():
    df = pd.read_csv(CONTROL_INPUT)

    required = {
        "method",
        "policy_seed",
        "split",
        "goal_angle",
        "mean_return",
    }
    missing = required - set(df.columns)
    assert not missing, f"Missing control columns: {sorted(missing)}"

    df = df[df["method"].isin(CONTROL_METHODS)].copy()

    assert not df.duplicated(
        ["method", "policy_seed", "goal_angle"]
    ).any()

    for method in CONTROL_METHODS:
        sub = df[df["method"] == method]

        seeds = sorted(sub["policy_seed"].astype(int).unique())
        assert seeds == [0, 1, 2, 3, 4], (method, seeds)

        angles = np.sort(sub["goal_angle"].astype(float).unique())
        assert np.allclose(angles, EXPECTED_ANGLES)

    return df


def aggregate_control(df):
    out = (
        df.groupby(["method", "split", "goal_angle"], as_index=False)
        .agg(
            mean_return=("mean_return", "mean"),
            std_return=("mean_return", lambda x: x.std(ddof=1)),
            seed_count=("policy_seed", "nunique"),
        )
        .sort_values(["method", "goal_angle"])
    )

    assert (out["seed_count"] == 5).all()
    return out


# ---------------------------------------------------------------------
# Probe data
# ---------------------------------------------------------------------

def load_probe():
    df = pd.read_csv(PROBE_INPUT)

    required = {
        "method",
        "encoder_seed",
        "split",
        "goal_angle",
        "circular_angle_mae",
        "fit_split",
    }
    missing = required - set(df.columns)
    assert not missing, f"Missing probe columns: {sorted(missing)}"

    df = df[df["method"].isin(LEARNED_METHODS)].copy()

    assert set(df["fit_split"]) == {"train"}

    assert not df.duplicated(
        ["method", "encoder_seed", "goal_angle"]
    ).any()

    for method in LEARNED_METHODS:
        sub = df[df["method"] == method]

        seeds = sorted(sub["encoder_seed"].astype(int).unique())
        assert seeds == [0, 1, 2, 3, 4], (method, seeds)

        angles = np.sort(sub["goal_angle"].astype(float).unique())
        assert np.allclose(angles, EXPECTED_ANGLES)

    return df


def aggregate_probe(df):
    out = (
        df.groupby(["method", "split", "goal_angle"], as_index=False)
        .agg(
            mean_mae=("circular_angle_mae", "mean"),
            std_mae=("circular_angle_mae", lambda x: x.std(ddof=1)),
            seed_count=("encoder_seed", "nunique"),
        )
        .sort_values(["method", "goal_angle"])
    )

    assert (out["seed_count"] == 5).all()
    return out


def load_state_only():
    df = pd.read_csv(
        STATE_INPUT,
        usecols=[
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

    # State-only rows are repeated bookkeeping rows, not independent seeds.
    nunique = (
        df.groupby("goal_angle")["angle_circular_angle_mae"]
        .nunique()
    )
    assert (nunique == 1).all()

    out = (
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
        .rename(
            columns={
                "angle_circular_angle_mae": "mean_mae"
            }
        )
    )

    assert len(out) == 13
    assert np.allclose(
        np.sort(out["goal_angle"].unique()),
        EXPECTED_ANGLES,
    )

    return out


# ---------------------------------------------------------------------
# Shared drawing helpers
# ---------------------------------------------------------------------

def decorate_regions(ax):
    ax.axvspan(
        -1.05,
        -0.60,
        color="0.94",
        zorder=0,
    )
    ax.axvspan(
        0.60,
        1.05,
        color="0.94",
        zorder=0,
    )

    ax.axvline(
        -0.60,
        color="0.7",
        linewidth=0.7,
        linestyle=":",
    )
    ax.axvline(
        0.60,
        color="0.7",
        linewidth=0.7,
        linestyle=":",
    )

    ax.grid(
        axis="y",
        linewidth=0.45,
        alpha=0.25,
    )

    ax.set_xlim(-1.05, 1.05)


def add_region_labels(ax):
    for x, label in [
        (-0.82, "OOD-L"),
        (0.00, "Train / ID"),
        (0.82, "OOD-R"),
    ]:
        ax.text(
            x,
            1.025,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=7.0,
            clip_on=False,
        )


def draw_control_series(ax, summary, method):
    d = (
        summary[summary["method"] == method]
        .sort_values("goal_angle")
    )

    st = STYLE[method]

    ax.errorbar(
        d["goal_angle"],
        d["mean_return"],
        yerr=d["std_return"],
        color=st["color"],
        linestyle=st["linestyle"],
        linewidth=st["linewidth"],
        marker=st["marker"],
        markersize=3.6,
        markeredgewidth=0.7,
        capsize=1.7,
        elinewidth=0.7,
        alpha=0.95,
        zorder=3,
    )


def draw_probe_series(ax, summary, method):
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
        markersize=3.6,
        markeredgewidth=0.7,
        capsize=1.7,
        elinewidth=0.7,
        alpha=0.95,
        zorder=3,
    )


# ---------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    control = aggregate_control(load_control())
    probe = aggregate_probe(load_probe())
    state = load_state_only()

    control.to_csv(
        OUTDIR / "main_figure_control_summary.csv",
        index=False,
    )
    probe.to_csv(
        OUTDIR / "main_figure_probe_summary.csv",
        index=False,
    )

    # Full-width two-panel figure.
    fig = plt.figure(figsize=(7.2, 3.05))

    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.05, 1.0],
        wspace=0.28,
    )

    # -------------------------------------------------------------
    # (a) Return panel with broken y-axis
    # -------------------------------------------------------------

    left = outer[0].subgridspec(
        2,
        1,
        height_ratios=[5.0, 0.85],
        hspace=0.06,
    )

    ax_ret = fig.add_subplot(left[0])
    ax_ret_low = fig.add_subplot(
        left[1],
        sharex=ax_ret,
    )

    for method in CONTROL_METHODS:
        draw_control_series(
            ax_ret,
            control,
            method,
        )
        draw_control_series(
            ax_ret_low,
            control,
            method,
        )

    decorate_regions(ax_ret)
    decorate_regions(ax_ret_low)

    ax_ret.set_ylim(-50, 1)
    ax_ret_low.set_ylim(-190, -140)

    ax_ret.spines["bottom"].set_visible(False)
    ax_ret_low.spines["top"].set_visible(False)

    ax_ret.tick_params(
        axis="x",
        which="both",
        bottom=False,
        labelbottom=False,
    )

    # Broken-axis marks.
    d = 0.012

    kwargs = dict(
        transform=ax_ret.transAxes,
        color="k",
        clip_on=False,
        linewidth=0.7,
    )
    ax_ret.plot(
        (-d, +d),
        (-d, +d),
        **kwargs,
    )
    ax_ret.plot(
        (1 - d, 1 + d),
        (-d, +d),
        **kwargs,
    )

    kwargs.update(
        transform=ax_ret_low.transAxes
    )
    ax_ret_low.plot(
        (-d, +d),
        (1 - d, 1 + d),
        **kwargs,
    )
    ax_ret_low.plot(
        (1 - d, 1 + d),
        (1 - d, 1 + d),
        **kwargs,
    )

    ax_ret.set_ylabel(
        "Mean episodic return",
        fontsize=8.5,
    )

    ax_ret_low.set_xlabel(
        r"Goal angle $\varphi$ (rad)",
        fontsize=8.5,
    )

    ticks = [
        -1.0,
        -0.6,
        -0.3,
        0.0,
        0.3,
        0.6,
        1.0,
    ]
    ax_ret_low.set_xticks(ticks)

    add_region_labels(ax_ret)

    control_handles = [
        Line2D(
            [0],
            [0],
            color=STYLE[m]["color"],
            linestyle=STYLE[m]["linestyle"],
            marker=STYLE[m]["marker"],
            linewidth=STYLE[m]["linewidth"],
            markersize=3.6,
            label=DISPLAY_NAME[m],
        )
        for m in CONTROL_METHODS
    ]

    ax_ret.legend(
        handles=control_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        frameon=False,
        fontsize=6.8,
        columnspacing=0.9,
        handlelength=1.8,
    )

    c_far = control[
        (control["method"] == "contrastive")
        & np.isclose(
            control["goal_angle"],
            1.0,
        )
    ].iloc[0]

    ax_ret_low.annotate(
        rf'${c_far["mean_return"]:.0f}'
        rf'\pm{c_far["std_return"]:.0f}$',
        xy=(
            1.0,
            c_far["mean_return"],
        ),
        xytext=(-7, -1),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=6.5,
    )

    ax_ret.text(
        -0.15,
        1.12,
        "(a) Control performance",
        transform=ax_ret.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
    )

    # -------------------------------------------------------------
    # (b) Probe panel
    # -------------------------------------------------------------

    ax_probe = fig.add_subplot(outer[1])

    decorate_regions(ax_probe)

    # State-only baseline.
    st = STYLE["state_only"]

    ax_probe.plot(
        state["goal_angle"],
        state["mean_mae"],
        color=st["color"],
        linestyle=st["linestyle"],
        linewidth=st["linewidth"],
        marker=st["marker"],
        markersize=3.6,
        markeredgewidth=0.7,
        alpha=0.95,
        zorder=2,
    )

    for method in LEARNED_METHODS:
        draw_probe_series(
            ax_probe,
            probe,
            method,
        )

    ax_probe.set_ylim(0.0, 1.05)

    ax_probe.set_ylabel(
        "Linear probe MAE (rad)",
        fontsize=8.5,
    )

    ax_probe.set_xlabel(
        r"Goal angle $\varphi$ (rad)",
        fontsize=8.5,
    )

    ax_probe.set_xticks(ticks)

    add_region_labels(ax_probe)

    probe_handles = [
        Line2D(
            [0],
            [0],
            color=STYLE[m]["color"],
            linestyle=STYLE[m]["linestyle"],
            marker=STYLE[m]["marker"],
            linewidth=STYLE[m]["linewidth"],
            markersize=3.6,
            label=DISPLAY_NAME[m],
        )
        for m in [
            "state_only",
            "vae",
            "contrastive",
        ]
    ]

    ax_probe.legend(
        handles=probe_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=3,
        frameon=False,
        fontsize=6.8,
        columnspacing=0.8,
        handlelength=1.8,
    )

    ax_probe.text(
        -0.15,
        1.12,
        "(b) Context decodability",
        transform=ax_probe.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
    )

    # -------------------------------------------------------------
    # Shared formatting
    # -------------------------------------------------------------

    for ax in [
        ax_ret,
        ax_ret_low,
        ax_probe,
    ]:
        ax.tick_params(
            labelsize=7.2,
        )

    fig.savefig(
        OUTDIR / "pointrobot_main_results.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        OUTDIR / "pointrobot_main_results.png",
        dpi=400,
        bbox_inches="tight",
    )

    print("Wrote:")
    print(
        OUTDIR
        / "pointrobot_main_results.pdf"
    )
    print(
        OUTDIR
        / "pointrobot_main_results.png"
    )


if __name__ == "__main__":
    main()
