from pathlib import Path
import subprocess
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

AGENTS = ["rnd_dqn", "rnd_ppo", "ppo", "dqn"]
SEEDS = [0, 1, 2]

def run_experiments():
    for agent in AGENTS:
        if agent == "rnd_dqn":
            path = "rnd_dqn.py"
        if agent == "rnd_ppo":
            path = "rnd_ppo.py"
        if agent == "ppo":
            path = "../week_6/ppo.py" 
        if agent == "dqn":
            path = "../week_4/dqn.py"
        for seed in SEEDS:
            print(f"Running: agent={agent}, seed={seed}")

            cmd = [
                "python", path,
                "-m",
                f"env.name=LunarLander-v3",
                f"seed={seed}",
            ]

            subprocess.run(cmd, check=True)

def load_results():
    base = Path("multirun")

    all_data = []

    paths = list(base.rglob("*.csv"))

    for path in paths:
        df = pd.read_csv(path)

        parts = path.parts

        agent = next((p for p in parts if p in AGENTS), "unknown")

        stem = path.stem
        seed = None
        for token in stem.split("_"):
            if token.isdigit():
                seed = int(token)
                break
        if seed is None:
            seed = -1

        df["agent"] = agent
        df["seed"] = seed
        df["env"] = "LunarLander-v3"

        all_data.append(df)

    if len(all_data) == 0:
        print("WARNING: no CSV log files found under searched paths  (multirun/)")
        return pd.DataFrame()

    return pd.concat(all_data, ignore_index=True)

def plot_training_curves_with_ci(df, env_name="LunarLander-v3"):
    env_df = df[df["env"] == env_name].copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    for agent in AGENTS:
        agent_df = env_df[env_df["agent"] == agent]

        if len(agent_df) == 0:
            print(f"WARNING: no data for agent {agent}")
            continue

        # Step × Seed Matrix
        pivot = agent_df.pivot_table(
            index="step",
            columns="seed",
            values="mean_return"
        ).sort_index()

        # Mean over seeds
        mean = pivot.mean(axis=1)

        # Confidence interval (95%)
        std = pivot.std(axis=1)
        n = pivot.count(axis=1)

        ci = 1.96 * std / np.sqrt(n)

        # Plot mean
        ax.plot(
            pivot.index,
            mean,
            label=agent
        )

        # Fill CI
        ax.fill_between(
            pivot.index,
            mean - ci,
            mean + ci,
            alpha=0.2
        )

    ax.set_title(f"Training Curves (Mean ± 95% CI) - {env_name}")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Return")

    ax.legend()
    fig.tight_layout()

    Path("plots").mkdir(exist_ok=True)
    fig.savefig(f"plots/{env_name}_training_curves.png",
                dpi=300, bbox_inches="tight")

    plt.close(fig)

def main():
    run_experiments()
    df = load_results()
    plot_training_curves_with_ci(df)

if __name__ == "__main__":
    main()