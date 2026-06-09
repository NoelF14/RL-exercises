from typing import Dict, List, Tuple

import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from rl_exercises.week_4.dqn import DQNAgent, set_seed
from rliable import library as rl_library
from rliable import metrics, plot_utils

# Define configurations to test
CONFIGS = [
    {
        "name": "baseline",
        "buffer_capacity": 10000,
        "batch_size": 32,
        "hidden_dim": 64,
        "num_hidden_layers": 2,
    },
    # Buffer capacity variations
    {
        "name": "small_buffer",
        "buffer_capacity": 5000,
        "batch_size": 32,
        "hidden_dim": 64,
        "num_hidden_layers": 2,
    },
    {
        "name": "large_buffer",
        "buffer_capacity": 20000,
        "batch_size": 32,
        "hidden_dim": 64,
        "num_hidden_layers": 2,
    },
    # Batch size variations
    {
        "name": "small_batch",
        "buffer_capacity": 10000,
        "batch_size": 16,
        "hidden_dim": 64,
        "num_hidden_layers": 2,
    },
    {
        "name": "large_batch",
        "buffer_capacity": 10000,
        "batch_size": 64,
        "hidden_dim": 64,
        "num_hidden_layers": 2,
    },
    # Network width variations
    {
        "name": "narrow_network",
        "buffer_capacity": 10000,
        "batch_size": 32,
        "hidden_dim": 32,
        "num_hidden_layers": 2,
    },
    {
        "name": "wide_network",
        "buffer_capacity": 10000,
        "batch_size": 32,
        "hidden_dim": 128,
        "num_hidden_layers": 2,
    },
    # Network depth variations
    {
        "name": "shallow_network",
        "buffer_capacity": 10000,
        "batch_size": 32,
        "hidden_dim": 64,
        "num_hidden_layers": 1,
    },
    {
        "name": "deep_network",
        "buffer_capacity": 10000,
        "batch_size": 32,
        "hidden_dim": 64,
        "num_hidden_layers": 4,
    },
]

SEEDS = [0, 1, 2, 3, 4, 42, 96]
#SEEDS = [0]
NUM_FRAMES = 20000
EVAL_INTERVAL = 100


def run_experiment(config: Dict, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    env = gym.make("CartPole-v1")
    set_seed(env, seed)

    agent = DQNAgent(
        env=env,
        buffer_capacity=config["buffer_capacity"],
        batch_size=config["batch_size"],
        hidden_dim=config["hidden_dim"],
        num_hidden_layers=config["num_hidden_layers"],
        seed=seed,
    )

    frames, rewards = agent.train_and_collect_rewards(NUM_FRAMES)
    env.close()
    return frames, rewards


def collect_all_results() -> Dict[str, Dict[int, np.ndarray]]:
    """
    Run all configurations across all seeds.

    Returns
    -------
    results : Dict
        results[config_name][seed] = array of episode rewards
    """
    results = {}

    for config in CONFIGS:
        print(f"\n{'=' * 50}")
        print(f"Config: {config['name']}")
        print(f"{'=' * 50}")

        results[config["name"]] = {}

        for seed in SEEDS:
            print(f"  Seed {seed}...", end=" ")
            frames, rewards = run_experiment(config, seed)
            results[config["name"]][seed] = {
                "frames": frames,
                "rewards": rewards,
            }
            print(f"✓ ({len(rewards)} episodes)")

    return results


CARTPOLE_MAX_REWARD = 500.0
BOOTSTRAP_REPS = 2000
FINAL_WINDOW_EPISODES = 100


def _build_frame_score_dict(results: Dict) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Convert variable-length per-seed episode logs into fixed frame-grid curves.

    Returns
    -------
    frame_grid:
        Shape: (num_checkpoints,)
    score_dict:
        config_name -> array of shape (num_seeds, 1, num_checkpoints)

    The singleton dimension is the 'num_tasks' dimension expected by RLiable.
    CartPole is one task, so num_tasks = 1.
    """
    frame_grid = np.arange(EVAL_INTERVAL, NUM_FRAMES + 1, EVAL_INTERVAL)
    score_dict = {}

    for config_name, seed_dict in results.items():
        seed_curves = []

        for seed in SEEDS:
            frames = np.asarray(seed_dict[seed]["frames"], dtype=np.float32)
            rewards = np.asarray(seed_dict[seed]["rewards"], dtype=np.float32)

            if len(frames) == 0 or len(rewards) == 0:
                curve = np.zeros_like(frame_grid, dtype=np.float32)
            else:
                curve = np.interp(
                    frame_grid,
                    frames,
                    rewards,
                    left=rewards[0],
                    right=rewards[-1],
                ).astype(np.float32)

            seed_curves.append(curve)

        # RLiable shape: (num_runs, num_tasks, num_frames)
        score_dict[config_name] = np.asarray(seed_curves, dtype=np.float32)[:, None, :]

    return frame_grid, score_dict


def _build_final_score_dict(
    results: Dict,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Build final-performance score dictionaries for RLiable.

    Returns
    -------
    raw_score_dict:
        config_name -> array of shape (num_seeds, 1), raw CartPole rewards.
    normalized_score_dict:
        config_name -> array of shape (num_seeds, 1), rewards / 500.
    """
    raw_score_dict = {}

    for config_name, seed_dict in results.items():
        final_scores = []

        for seed in SEEDS:
            rewards = np.asarray(seed_dict[seed]["rewards"], dtype=np.float32)

            if len(rewards) == 0:
                final_score = 0.0
            else:
                final_score = float(np.mean(rewards[-FINAL_WINDOW_EPISODES:]))

            final_scores.append(final_score)

        raw_scores = np.asarray(final_scores, dtype=np.float32)[:, None]
        raw_score_dict[config_name] = raw_scores

    normalized_score_dict = {
        name: np.clip(scores / CARTPOLE_MAX_REWARD, 0.0, 1.0)
        for name, scores in raw_score_dict.items()
    }

    return raw_score_dict, normalized_score_dict


def _plot_interval_points(
    ax,
    point_estimates: Dict[str, np.ndarray],
    interval_estimates: Dict[str, np.ndarray],
    metric_names: List[str],
    algorithms: List[str],
    title: str,
    xlabel: str,
) -> None:
    """
    Plot RLiable point estimates and bootstrap CIs on a provided axis.
    """
    y = np.arange(len(algorithms))
    num_metrics = len(metric_names)

    if num_metrics == 1:
        offsets = np.array([0.0])
    else:
        offsets = np.linspace(-0.25, 0.25, num_metrics)

    for metric_idx, metric_name in enumerate(metric_names):
        points = np.array(
            [point_estimates[algo][metric_idx] for algo in algorithms],
            dtype=np.float32,
        )

        lower = np.array(
            [interval_estimates[algo][0, metric_idx] for algo in algorithms],
            dtype=np.float32,
        )
        upper = np.array(
            [interval_estimates[algo][1, metric_idx] for algo in algorithms],
            dtype=np.float32,
        )

        xerr = np.vstack([points - lower, upper - points])

        ax.errorbar(
            points,
            y + offsets[metric_idx],
            xerr=xerr,
            fmt="o",
            capsize=3,
            label=metric_name,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(algorithms)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="x")

    if num_metrics > 1:
        ax.legend()

def plot_l1_training_curves(results: Dict) -> None:
    """
    Simple L1 plots:
    Reward across frames.
    """
    os.makedirs("results", exist_ok=True)

    frame_grid = np.arange(EVAL_INTERVAL, NUM_FRAMES + 1, EVAL_INTERVAL)

    plt.figure(figsize=(12, 8))

    for config_name, seed_dict in results.items():
        frames = np.asarray(seed_dict[0]["frames"], dtype=np.float32)
        rewards = np.asarray(seed_dict[0]["rewards"], dtype=np.float32)

        # Interpolate onto common frame grid
        curve = np.interp(
            frame_grid,
                frames,
                rewards,
                left=rewards[0],
                right=rewards[-1],
        )

        plt.plot(
            frame_grid,
            curve,
            label=config_name,
        )

    plt.xlabel("Frames")
    plt.ylabel("Reward")
    plt.title("DQN Training Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/dqn_training_curves_l1.png", dpi=150)
    plt.close()


def plot_with_rliable(results: Dict) -> None:
    """
    Use RLiable to plot:
      1. IQM training curve over frames with bootstrap CIs.
      2. Final IQM / median / mean with bootstrap CIs.
      3. Final optimality gap with bootstrap CIs.
      4. Performance profile over normalized final reward thresholds.
    """
    os.makedirs("results", exist_ok=True)

    config_names = list(results.keys())

    frame_grid, frame_score_dict = _build_frame_score_dict(results)
    raw_final_score_dict, normalized_final_score_dict = _build_final_score_dict(results)

    def iqm_over_time(scores: np.ndarray) -> np.ndarray:
        return np.array(
            [
                metrics.aggregate_iqm(scores[..., frame_idx])
                for frame_idx in range(scores.shape[-1])
            ]
        )

    def aggregate_reward_metrics(scores: np.ndarray) -> np.ndarray:
        return np.array(
            [
                metrics.aggregate_iqm(scores),
                metrics.aggregate_median(scores),
                metrics.aggregate_mean(scores),
            ]
        )

    def optimality_gap(scores: np.ndarray) -> np.ndarray:
        return np.array([metrics.aggregate_optimality_gap(scores)])

    print("Computing IQM training curve CIs...")
    iqm_scores, iqm_cis = rl_library.get_interval_estimates(
        frame_score_dict,
        iqm_over_time,
        reps=BOOTSTRAP_REPS,
    )

    print("Computing aggregate metric CIs...")
    aggregate_scores, aggregate_cis = rl_library.get_interval_estimates(
        raw_final_score_dict,
        aggregate_reward_metrics,
        reps=BOOTSTRAP_REPS,
    )

    print("Computing optimality gap CIs...")
    gap_scores, gap_cis = rl_library.get_interval_estimates(
        normalized_final_score_dict,
        optimality_gap,
        reps=BOOTSTRAP_REPS,
    )

    print("Computing performance profile CIs...")
    thresholds = np.linspace(0.0, 1.0, 51)

    score_distributions, score_distribution_cis = rl_library.create_performance_profile(
        normalized_final_score_dict,
        thresholds,
        reps=BOOTSTRAP_REPS,
    )

    # IMPORTANT: axes must be created before using axes[0, 0], axes[1, 1], etc.
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))

    # 1. IQM training curve over frames
    plot_utils.plot_sample_efficiency_curve(
        frame_grid,
        iqm_scores,
        iqm_cis,
        algorithms=config_names,
        xlabel="Frames",
        ylabel="IQM rolling episode reward",
        ax=axes[0, 0],
        marker=None,
        labelsize="medium",
        ticklabelsize="small",
        legend=True,
        legendsize="small",
    )
    axes[0, 0].set_title("Training curve: IQM across seeds with 95% CI")

    # 2. Final IQM / median / mean
    _plot_interval_points(
        axes[0, 1],
        aggregate_scores,
        aggregate_cis,
        metric_names=["IQM", "Median", "Mean"],
        algorithms=config_names,
        title=f"Final reward metrics, last {FINAL_WINDOW_EPISODES} episodes",
        xlabel="CartPole reward",
    )

    # 3. Optimality gap
    _plot_interval_points(
        axes[1, 0],
        gap_scores,
        gap_cis,
        metric_names=["Optimality Gap"],
        algorithms=config_names,
        title="Optimality gap to max CartPole score",
        xlabel="Normalized gap, lower is better",
    )

    # 4. Performance profile
    plot_utils.plot_performance_profiles(
        score_distributions,
        thresholds,
        performance_profile_cis=score_distribution_cis,
        xlabel=r"Normalized final reward threshold $(\tau)$",
        ylabel=r"Fraction of runs with score $> \tau$",
        ax=axes[1, 1],
        labelsize="medium",
        ticklabelsize="small",
        legend=True,
        legendsize="small",
    )
    axes[1, 1].set_title("Performance profile")

    plt.tight_layout()
    plt.savefig("results/dqn_robust_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()

def write_observations_l1(results: Dict) -> None:
    with open("observations_l1.txt", "w") as f:
        f.write("L1 DQN Hyperparameter Study\n")
        f.write("=" * 50 + "\n\n")

        for config_name, seed_dict in results.items():

            rewards = np.asarray(seed_dict[0]["rewards"])
            final_reward = np.mean(rewards[-100:])

            f.write(f"Configuration: {config_name}\n")
            f.write(f"final reward: {final_reward:.2f}\n\n")

        f.write("\nObservations:\n")

def write_observations(results: Dict) -> None:
    """
    Write observations to observations_l2.txt.
    """
    raw_final_score_dict, normalized_final_score_dict = _build_final_score_dict(results)

    with open("observations_l2.txt", "w") as f:
        f.write("DQN Ablation Study with Robust Reporting (RLiable)\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Seeds: {SEEDS}\n")
        f.write(f"Training frames per run: {NUM_FRAMES}\n")
        f.write(f"Final score window: last {FINAL_WINDOW_EPISODES} episodes\n")
        f.write(f"CartPole normalization: reward / {CARTPOLE_MAX_REWARD}\n\n")

        for config_name in sorted(results.keys()):
            raw_scores = raw_final_score_dict[config_name][:, 0]
            normalized_scores = normalized_final_score_dict[config_name]

            iqm = metrics.aggregate_iqm(raw_scores[:, None])
            median = metrics.aggregate_median(raw_scores[:, None])
            mean = metrics.aggregate_mean(raw_scores[:, None])
            opt_gap = metrics.aggregate_optimality_gap(normalized_scores)

            f.write(f"Configuration: {config_name}\n")
            f.write("-" * 40 + "\n")
            f.write(f"  IQM reward:              {iqm:.2f}\n")
            f.write(f"  Median reward:           {median:.2f}\n")
            f.write(f"  Mean reward:             {mean:.2f}\n")
            f.write(f"  Std reward:              {raw_scores.std():.2f}\n")
            f.write(
                f"  Min/Max reward:          {raw_scores.min():.2f} / {raw_scores.max():.2f}\n"
            )
            f.write(f"  Optimality gap:          {opt_gap:.4f}\n")
            f.write(f"  Per-seed final rewards:  {np.round(raw_scores, 2)}\n")
            f.write("\n")

        f.write("\nObservations:\n")
        f.write("-" * 40 + "\n")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    print("Collecting DQN results across configurations and seeds...")
    results = collect_all_results()

    print("Plot l1 results")
    plot_l1_training_curves(results)
    write_observations_l1(results)

    print("Generating RLiable plots...")
    plot_with_rliable(results)

    print("Writing observations...")
    write_observations(results)

