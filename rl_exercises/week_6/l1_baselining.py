import subprocess
import itertools
import os
from pathlib import Path
import pandas as pd
import numpy as np

from rliable import library as rly
from rliable import metrics
from rliable import plot_utils
import matplotlib.pyplot as plt


BASELINES = ["none", "avg", "value", "gae"]
ENVS = ["CartPole-v1", "LunarLander-v3"]
SEEDS = [0, 1, 2]

def run_experiments():
    for env, baseline, seed in itertools.product(ENVS, BASELINES, SEEDS):
        print(f"Running: env={env}, baseline={baseline}, seed={seed}")

        cmd = [
            "python", "actor_critic.py",
            "-m",
            f"env.name={env}",
            f"agent.baseline_type={baseline}",
            f"seed={seed}",
        ]

        subprocess.run(cmd, check=True)

def load_results():
    # also check Hydra multirun layout
    base = Path("multirun")

    all_data = []

    paths = list(base.rglob("*.csv"))

    for path in paths:
        df = pd.read_csv(path)

        parts = path.parts

        baseline = next((p for p in parts if p in BASELINES), "unknown")
        env = next((p for p in parts if any(e in p for e in ENVS)), "unknown")

        # try to extract seed from filename like seed_0.csv
        stem = path.stem
        seed = None
        for token in stem.split("_"):
            if token.isdigit():
                seed = int(token)
                break
        if seed is None:
            seed = -1

        df["baseline"] = baseline
        df["env"] = env
        df["seed"] = seed

        all_data.append(df)

    if len(all_data) == 0:
        print("WARNING: no CSV log files found under searched paths  (multirun/)")
        return pd.DataFrame()

    return pd.concat(all_data, ignore_index=True)

def plot_training_curves_with_ci(df, env_name, BASELINES):
    env_df = df[df["env"] == env_name].copy()
    env_df = env_df.dropna(subset=["eval_return_mean"])

    fig, ax = plt.subplots(figsize=(10, 6))

    for baseline in BASELINES:
        baseline_df = env_df[env_df["baseline"] == baseline]

        if len(baseline_df) == 0:
            print(f"WARNING: no data for baseline {baseline}")
            continue

        # Step × Seed Matrix
        pivot = baseline_df.pivot_table(
            index="step",
            columns="seed",
            values="eval_return_mean"
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
            label=baseline
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
    fig.savefig(f"plots/{env_name}_training_curves_ci.png",
                dpi=300, bbox_inches="tight")

    plt.close(fig)

def main():
    run_experiments()

    df = load_results()

    for env in ENVS:
        plot_training_curves_with_ci(df, env, BASELINES)

if __name__ == "__main__":
    main()