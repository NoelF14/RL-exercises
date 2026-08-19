from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT = Path(
    "results/cartpole_force_encoders/analysis/"
    "cartpole_force_latent_probe_5seed_by_seed.csv"
)

OUTPUT = Path("figures")
OUTPUT.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(INPUT)

ORDER = [
    "state_only",
    "raw_history_H5",
    "vae",
    "contrastive",
]

LABELS = {
    "state_only": "State only",
    "raw_history_H5": r"Raw history ($H=5$)",
    "vae": "VAE latent",
    "contrastive": "Contrastive latent",
}


rows = []

for method in ORDER:
    block = df[
        df["method"] == method
    ]

    if method in {
        "state_only",
        "raw_history_H5",
    }:
        row = {
            "method": method,
            "mae": float(block["mae"].iloc[0]),
            "mae_sd": 0.0,
            "r2": float(block["r2"].iloc[0]),
            "r2_sd": 0.0,
        }
    else:
        row = {
            "method": method,
            "mae": float(block["mae"].mean()),
            "mae_sd": float(block["mae"].std()),
            "r2": float(block["r2"].mean()),
            "r2_sd": float(block["r2"].std()),
        }

    rows.append(row)


summary = pd.DataFrame(rows)

y = np.arange(len(ORDER))


fig, axes = plt.subplots(
    1,
    2,
    figsize=(7.2, 2.8),
    sharey=True,
)


# ------------------------------------------------------------
# MAE
# ------------------------------------------------------------

axes[0].barh(
    y,
    summary["mae"],
    xerr=summary["mae_sd"],
    capsize=2,
)

axes[0].set_xlabel("Force probe MAE")
axes[0].set_yticks(y)
axes[0].set_yticklabels(
    [LABELS[m] for m in ORDER]
)
axes[0].invert_yaxis()
axes[0].grid(
    axis="x",
    alpha=0.25,
    linewidth=0.7,
)


# ------------------------------------------------------------
# R^2
# ------------------------------------------------------------

axes[1].barh(
    y,
    summary["r2"],
    xerr=summary["r2_sd"],
    capsize=2,
)

axes[1].axvline(
    0.0,
    linewidth=0.8,
)

axes[1].set_xlabel(r"Force probe $R^2$")
axes[1].set_xlim(-0.05, 1.05)
axes[1].grid(
    axis="x",
    alpha=0.25,
    linewidth=0.7,
)


fig.tight_layout()

for suffix in ["pdf", "png"]:
    path = OUTPUT / f"cartpole_force_probe_5seed.{suffix}"
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )
    print(path)

plt.close(fig)
