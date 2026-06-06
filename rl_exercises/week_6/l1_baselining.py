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

def load_results(log_dir="multirun"):
    all_data = []

    for path in Path(log_dir).rglob("*.csv"):
        df = pd.read_csv(path)

        parts = path.parts

        baseline = next((p for p in parts if p in BASELINES), "unknown")
        env = next((p for p in parts if "CartPole" in p or "LunarLander" in p), "unknown")
        seed = int(path.stem.split("_")[-1])  # seed_0.csv

        df["baseline"] = baseline
        df["env"] = env
        df["seed"] = seed

        all_data.append(df)

    return pd.concat(all_data, ignore_index=True)

def to_rliable_format(df, env_name):
    results = {}

    env_df = df[df["env"] == env_name]

    for baseline in BASELINES:
        runs = []

        for seed in SEEDS:
            run = env_df[
                (env_df["baseline"] == baseline) &
                (env_df["seed"] == seed)
            ].sort_values("step")

            runs.append(run["eval_return_mean"].dropna().values)

        results[baseline] = np.array(runs)

    return results

def plot_rliable(results, df, env_name):
    aggregate_func = lambda x: np.array([
        metrics.aggregate_median(x),
        metrics.aggregate_iqm(x),
    ])

    scores, score_cis = rly.get_interval_estimates(
        results,
        aggregate_func,
        reps=2000
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    plot_utils.plot_interval_estimates(
        scores,
        score_cis,
        metric_names=["Median", "IQM"],
        ax=ax
    )

    ax.set_title(f"Final Performance - {env_name}")

    Path("plots").mkdir(exist_ok=True)
    fig.savefig(f"plots/{env_name}_baselines.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))

    env_df = df[df["env"] == env_name]

    for baseline in BASELINES:
        baseline_df = env_df[env_df["baseline"] == baseline]

        eval_df = baseline_df.dropna(subset=["eval_return_mean"])

        if len(eval_df) == 0:
            print(f"WARNING: no eval data for {baseline} in {env_name}")
            continue

        ax.plot(
            eval_df["step"],
            eval_df["eval_return_mean"],
            label=baseline
        )

    ax.set_title(f"Training Curves - {env_name}")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Return")

    ax.legend()
    fig.tight_layout()

    fig.savefig(f"plots/{env_name}_training_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    run_experiments()

    df = load_results()

    for env in ENVS:
        results = to_rliable_format(df, env)
        plot_rliable(results, df, env)

if __name__ == "__main__":
    main()