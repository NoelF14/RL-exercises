from __future__ import annotations

from rich import print as printr
from rl_exercises.environments import ContextualMarsRover

# convex region in feature space (mode A)
train_contexts_mode_a = [
    {"slip": s, "reward": r} for s in [0.0, 0.1, 0.2] for r in [8, 10, 12]
]

# small variation (mode b)
train_contexts_mode_b = [{"slip": s, "reward": r} for s in [0.1, 0.15] for r in [9, 10]]

# single values per feature, combinatorial interpolation (mode c)
train_contexts_mode_c = [
    {"slip": 0.1, "reward": 10},
    {"slip": 0.2, "reward": 10},
    {"slip": 0.1, "reward": 12},
]

# interpolation
validation_contexts = [
    {"slip": 0.15, "reward": 9},
    {"slip": 0.05, "reward": 11},
]

# extrapolation
test_contexts = [
    {"slip": 0.3, "reward": 10},  # new slip
    {"slip": 0.1, "reward": 6},  # new reward
    {"slip": 0.3, "reward": 6},  # both new
]


def run_episode(env: ContextualMarsRover, actions: list[int]) -> float:
    obs, info = env.reset()
    total_reward = 0.0

    for step, action in enumerate(actions):
        obs_next, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        printr(
            f"step={step}, obs={obs}, action={action}, "
            f"next_obs={obs_next}, reward={reward}, "
            f"terminated={terminated}, truncated={truncated}, info={info}"
        )

        obs = obs_next

        if terminated or truncated:
            break

    return total_reward


if __name__ == "__main__":
    actions = [0, 0, 1, 1, 1, 1, 1]

    train_contexts = train_contexts_mode_a
    for episode_idx in range(6):
        context = train_contexts[episode_idx % len(train_contexts)]

        env = ContextualMarsRover(
            context=context,
            expose_context=True,
            horizon=10,
            seed=episode_idx,
        )

        printr(f"\n[bold]Episode {episode_idx}[/bold]")
        printr(f"context={context}")

        total_reward = run_episode(env, actions)
        printr(f"total_reward={total_reward}")
