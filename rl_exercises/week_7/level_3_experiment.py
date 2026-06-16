from email.mime import base
from pathlib import Path
import subprocess
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def load_results():
    path_1 = Path("outputs/2026-06-11/15-58-54/dqn_ensemble/seed_0.csv")
    path_2 = Path("outputs/2026-06-11/16-35-34/rnd_dqn/seed_0.csv")

    df_1 = pd.read_csv(path_1)
    df_1["agent"] = "dqn_ensemble"

    df_2 = pd.read_csv(path_2)
    df_2["agent"] = "rnd_dqn"

    return pd.concat([df_1, df_2], ignore_index=True)

def plot_training_curves(data):
    fig, ax = plt.subplots(figsize=(10, 6))

    for agent in ["rnd_dqn", "dqn_ensemble"]:
        agent_df = data[data["agent"] == agent]

        ax.plot(
            agent_df["step"],
            agent_df["mean_return"],
            label=agent
        )

    ax.set_title(f"Training Curves - Level_3")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Return")

    ax.legend()
    fig.tight_layout()

    Path("plots").mkdir(exist_ok=True)
    fig.savefig(f"plots/Level_3_training_curves.png",
                dpi=300, bbox_inches="tight")

    plt.close(fig)

def main():
    data = load_results()
    plot_training_curves(data)

if __name__ == "__main__":
    main()