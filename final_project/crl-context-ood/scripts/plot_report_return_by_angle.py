from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


INPUT = Path("results/pointrobot_primary/primary_context_results.csv")
OUTDIR = Path("figures")

EXPECTED_METHODS = ["no_context", "oracle", "vae", "contrastive"]
EXPECTED_ANGLES = np.array(
    [-1.0, -0.8, -0.6, -0.45, -0.3, -0.15, 0.0,
      0.15, 0.3, 0.45, 0.6, 0.8, 1.0]
)

DISPLAY_NAME = {
    "oracle": "Oracle",
    "no_context": "No context",
    "vae": "VAE",
    "contrastive": "Contrastive",
}

# Keep these styles identical when we build the probe panel.
STYLE = {
    "oracle": {
        "color": "black",
        "marker": "o",
        "linestyle": "-",
        "linewidth": 1.4,
    },
    "no_context": {
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

PLOT_ORDER = ["oracle", "no_context", "vae", "contrastive"]


def load_and_validate():
    df = pd.read_csv(INPUT)

    required = {
        "method",
        "policy_seed",
        "split",
        "goal_angle",
        "mean_return",
    }
    missing = required - set(df.columns)
    assert not missing, f"Missing columns: {sorted(missing)}"

    df = df[df["method"].isin(EXPECTED_METHODS)].copy()

    assert set(df["method"]) == set(EXPECTED_METHODS)

    # Exactly one context-level result per method / policy seed / angle.
    assert not df.duplicated(
        ["method", "policy_seed", "goal_angle"]
    ).any()

    for method in EXPECTED_METHODS:
        sub = df[df["method"] == method]

        seeds = sorted(sub["policy_seed"].astype(int).unique())
        assert seeds == [0, 1, 2, 3, 4], (method, seeds)

        angles = np.sort(sub["goal_angle"].astype(float).unique())
        assert np.allclose(angles, EXPECTED_ANGLES), (method, angles)

    expected_split = {
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

    for angle, split in expected_split.items():
        actual = set(
            df.loc[
                np.isclose(df["goal_angle"].astype(float), angle),
                "split",
            ]
        )
        assert actual == {split}, (angle, actual)

    return df


def aggregate(df):
    summary = (
        df.groupby(["method", "split", "goal_angle"], as_index=False)
        .agg(
            mean_return=("mean_return", "mean"),
            std_return=("mean_return", lambda x: x.std(ddof=1)),
            seed_count=("policy_seed", "nunique"),
        )
        .sort_values(["method", "goal_angle"])
    )

    assert (summary["seed_count"] == 5).all()
    return summary


def draw_series(ax, summary, method):
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
        markersize=4.0,
        markeredgewidth=0.8,
        capsize=2.0,
        elinewidth=0.8,
        alpha=0.95,
        label=DISPLAY_NAME[method],
        zorder=3,
    )


def decorate_context_regions(ax):
    # Light shading for extrapolation regions.
    ax.axvspan(-1.05, -0.60, color="0.94", zorder=0)
    ax.axvspan(0.60, 1.05, color="0.94", zorder=0)

    # Boundaries of the training range.
    ax.axvline(-0.60, color="0.7", linewidth=0.8, linestyle=":")
    ax.axvline(0.60, color="0.7", linewidth=0.8, linestyle=":")

    ax.grid(axis="y", linewidth=0.5, alpha=0.25)
    ax.set_xlim(-1.05, 1.05)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    df = load_and_validate()
    summary = aggregate(df)

    # Persist the exact values behind the report figure.
    summary.to_csv(
        OUTDIR / "pointrobot_return_by_angle_summary.csv",
        index=False,
    )

    # Slightly shorter figure so it will combine well with the probe panel.
    fig = plt.figure(figsize=(5.2, 3.15))

    # Broken y-axis:
    # main performance range above, catastrophic contrastive +1.0 below.
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[5.0, 0.85],
        hspace=0.06,
    )

    ax = fig.add_subplot(gs[0])
    ax_low = fig.add_subplot(gs[1], sharex=ax)

    for method in PLOT_ORDER:
        draw_series(ax, summary, method)
        draw_series(ax_low, summary, method)

    decorate_context_regions(ax)
    decorate_context_regions(ax_low)

    # Contains all means except catastrophic contrastive far-right.
    ax.set_ylim(-50, 1)

    # Explicit lower window for catastrophic far-right result.
    ax_low.set_ylim(-190, -140)

    # Broken-axis styling.
    ax.spines["bottom"].set_visible(False)
    ax_low.spines["top"].set_visible(False)

    ax.tick_params(
        axis="x",
        which="both",
        bottom=False,
        labelbottom=False,
    )

    d = 0.010

    kwargs = dict(
        transform=ax.transAxes,
        color="k",
        clip_on=False,
        linewidth=0.8,
    )
    ax.plot((-d, +d), (-d, +d), **kwargs)
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs.update(transform=ax_low.transAxes)
    ax_low.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_low.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    ax.set_ylabel("Mean episodic return")
    ax_low.set_xlabel(r"Goal angle $\varphi$ (rad)")

    # Deliberately sparse tick labels for paper readability.
    major_ticks = [-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0]
    ax_low.set_xticks(major_ticks)

    # Region labels above the plotting area so they do not overlap curves.
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

    # Compact method legend.
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
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        frameon=False,
        fontsize=8,
        columnspacing=1.2,
        handlelength=2.0,
    )

    # Explicit annotation for catastrophic far-right contrastive result.
    c_far = summary[
        (summary["method"] == "contrastive")
        & np.isclose(summary["goal_angle"], 1.0)
    ].iloc[0]

    ax_low.annotate(
        rf'${c_far["mean_return"]:.0f}\pm{c_far["std_return"]:.0f}$',
        xy=(1.0, c_far["mean_return"]),
        xytext=(-10, -1),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=7,
    )

    fig.savefig(
        OUTDIR / "pointrobot_return_by_angle.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        OUTDIR / "pointrobot_return_by_angle.png",
        dpi=400,
        bbox_inches="tight",
    )

    print(summary.to_string(index=False))
    print()
    print("Wrote:")
    print(OUTDIR / "pointrobot_return_by_angle.pdf")
    print(OUTDIR / "pointrobot_return_by_angle.png")
    print(OUTDIR / "pointrobot_return_by_angle_summary.csv")


if __name__ == "__main__":
    main()
