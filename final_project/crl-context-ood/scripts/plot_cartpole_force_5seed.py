from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT = Path(
    "results/cartpole_force_encoders/analysis/"
    "cartpole_force_5seed_context_returns.csv"
)

OUTPUT = Path("figures")
OUTPUT.mkdir(parents=True, exist_ok=True)


METHOD_ORDER = [
    "hidden",
    "oracle_raw",
    "oracle_clipped",
    "vae_history",
    "contrastive_history",
]

LABELS = {
    "hidden": "Hidden",
    "oracle_raw": "Oracle",
    "oracle_clipped": "Oracle clipped",
    "vae_history": "VAE",
    "contrastive_history": "Contrastive",
}

MARKERS = {
    "hidden": "o",
    "oracle_raw": "s",
    "oracle_clipped": "^",
    "vae_history": "D",
    "contrastive_history": "*",
}


df = pd.read_csv(INPUT)

summary = (
    df.groupby(
        ["method", "split", "force_mag"],
        as_index=False,
    )
    .agg(
        mean_return=("mean_return", "mean"),
        sd_return=("mean_return", "std"),
    )
)


fig, axes = plt.subplots(
    1,
    2,
    figsize=(7.2, 3.0),
    sharey=True,
)


for ax, split, title in [
    (axes[0], "ood_low", "Low-force OOD"),
    (axes[1], "ood_high", "High-force OOD"),
]:
    for method in METHOD_ORDER:
        block = (
            summary[
                (summary["method"] == method)
                & (summary["split"] == split)
            ]
            .sort_values("force_mag")
        )

        ax.errorbar(
            block["force_mag"],
            block["mean_return"],
            yerr=block["sd_return"],
            marker=MARKERS[method],
            linewidth=1.5,
            markersize=5,
            capsize=2,
            label=LABELS[method],
        )

    ax.set_title(title)
    ax.set_xlabel(r"Force magnitude $F$")
    ax.set_ylim(0, 525)
    ax.grid(
        axis="y",
        alpha=0.25,
        linewidth=0.7,
    )


axes[0].set_ylabel("Episode return")

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.03),
    ncol=5,
    frameon=False,
)

fig.tight_layout(
    rect=(0, 0.10, 1, 1)
)

for suffix in ["pdf", "png"]:
    path = OUTPUT / f"cartpole_force_ood_5seed.{suffix}"
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )
    print(path)

plt.close(fig)
