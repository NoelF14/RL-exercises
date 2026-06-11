"""
Deep Q-Learning with deep ensemble for exploration.
"""

import csv
import os
from typing import Any, Dict, List, Tuple
from hydra.core.hydra_config import HydraConfig

import gymnasium as gym
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from rl_exercises.agent import AbstractAgent
from rl_exercises.week_4.buffers import ReplayBuffer # noqa: F401
from torch import nn  # noqa: F401
import torch.nn.functional as F

def set_seed(env: gym.vector.VectorEnv, seed: int = 0) -> None:
    """
    Seed a vectorized Gym environment properly.
    """

    np.random.seed(seed)
    torch.manual_seed(seed)

    num_envs = env.num_envs

    seeds = [seed + i for i in range(num_envs)]

    env.reset(seed=seeds)

    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)

    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)

def quantile_loss(pred, target, taus):

    pred = pred.unsqueeze(2)      # [B,N,1]
    target = target.unsqueeze(1)  # [B,1,N]

    diff = target - pred

    abs_diff = diff.abs()

    huber = torch.where(
        abs_diff <= 1.0,
        0.5 * diff.pow(2),
        abs_diff - 0.5
    )

    taus = taus.view(1, -1, 1)

    loss = (
        torch.abs(
            taus - (diff.detach() < 0).float()
        )
        * huber
    ).mean()

    return loss

class FeatureExtractor(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
class QuantileQHead(nn.Module):
    def __init__(self, n_actions: int, n_quantiles: int, hidden_dim: int = 128):
        super().__init__()

        self.n_actions = n_actions
        self.n_quantiles = n_quantiles

        self.net = nn.Linear(hidden_dim, n_actions * n_quantiles)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]

        out = self.net(z)

        out = out.view(B, self.n_actions, self.n_quantiles)

        return out

class DQNEnsembleAgent(AbstractAgent):
    """
    Deep Q-Learning agent with deep ensemble for exploration.

    Derives from AbstractAgent by implementing:
      - predict_action
      - save / load
      - update_agent
    """

    def __init__(
        self,
        envs: gym.vector.VectorEnv,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        lr: float = 1e-3,
        gamma: float = 0.99,
        target_update_freq: int = 1000,
        seed: int = 0,
        heads: int = 5,
        K: int = 3,
        phi: float = 1.0,
        alpha: float = 1.0,
        lmbda: float = 0.5,
        N: int = 50
    ) -> None:
        """
        Initialize replay buffer, Q-networks, optimizer, and hyperparameters.

        Parameters
        ----------
        envs : gym.VectorEnv 
            The Gym environment.
        buffer_capacity : int
            Max experiences stored.
        batch_size : int
            Mini-batch size for updates.
        lr : float
            Learning rate.
        gamma : float
            Discount factor.
        epsilon_start : float
            Initial ε for exploration.
        epsilon_final : float
            Final ε.
        epsilon_decay : int
            Exponential decay parameter.
        target_update_freq : int
            How many updates between target-network syncs.
        seed : int
            RNG seed.
        heads : int
            Number of Heads in the ensemble.
        K : int
            Number of actors.
        phi : float
            UCB exploration coefficient.
        alpha : float
            Hyperparameter for controlling the shape of the coefficient distribution.
        lmbda : float
            Hyperparameter for controlling the shape of the coefficient distribution.
        N : int
            Number of quantiles for distributional RL.
        """
        super().__init__(
            envs,
            buffer_capacity,
            batch_size,
            lr,
            gamma,
            target_update_freq,
            seed,
            heads,
            K,
            phi,
            alpha,
            lmbda,
            N
        )
        self.K = K

        self.envs = envs

        self.seed = seed
        set_seed(self.envs, self.seed)

        obs_dim = envs.single_observation_space.shape[0]
        n_actions = envs.single_action_space.n
        self.heads = heads

        self.f = FeatureExtractor(obs_dim)
        self.g = nn.ModuleList([QuantileQHead(n_actions, N) for _ in range(self.heads)])

        self.f_target = FeatureExtractor(obs_dim)
        self.g_target = nn.ModuleList([QuantileQHead(n_actions, N) for _ in range(self.heads)])

        self.f_target.load_state_dict(self.f.state_dict())
        for i in range(self.heads):
            self.g_target[i].load_state_dict(self.g[i].state_dict())

        self.optimizer = torch.optim.Adam(
            list(self.f.parameters()) +
            [p for head in self.g for p in head.parameters()],
            lr=lr   
        )
        self.buffer = ReplayBuffer(buffer_capacity)

        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_freq = target_update_freq

        self.alpha = alpha
        self.lmbda = lmbda
        self.phi = phi
        self.N = N
        self.taus = (torch.arange(N) + 0.5) / N

        self.total_steps = 0

    def predict_action(self, states):
        states = torch.tensor(states, dtype=torch.float32)  

        with torch.no_grad():
            z = self.f(states)  
            q = torch.stack([h(z) for h in self.g])  

            #Q mean over heads + quantiles
            q_mean = q.mean(dim=(0, -1))  

            #epistemic uncertainty 
            head_means = q.mean(dim=-1) 
            sigma = head_means.var(dim=0).sqrt() 

            actions = []

            for k in range(self.K):
                phi_k = self.phi * self.lmbda + (k / (self.K - 1)) * self.alpha

                ucb = q_mean[k] + phi_k * sigma[k]  

                actions.append(torch.argmax(ucb).item())

        return actions
    
    def save(self, path: str) -> None:
        """
        Save model & optimizer state to disk.

        Parameters
        ----------
        path : str
            File path.
        """
        torch.save(
            {
                "f": self.f.state_dict(),
                "g": [head.state_dict() for head in self.g],
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """
        Load model & optimizer state from disk.

        Parameters
        ----------
        path : str
            File path.
        """
        checkpoint = torch.load(path)
        self.f.load_state_dict(checkpoint["f"])
        for i, head in enumerate(self.g):
            head.load_state_dict(checkpoint["g"][i])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def update_agent(self, batch):
        states, actions, rewards, next_states, dones, _ = zip(*batch)

        s = torch.tensor(np.array(states), dtype=torch.float32)
        a = torch.tensor(actions, dtype=torch.long)
        r = torch.tensor(rewards, dtype=torch.float32)
        s_next = torch.tensor(np.array(next_states), dtype=torch.float32)
        done = torch.tensor(dones, dtype=torch.float32)

        z = self.f(s)
        z_next = self.f_target(s_next)

        losses = []
        total_loss = 0

        for i in range(self.heads):
            dist = self.g[i](z)

            # action selection → take all quantiles of chosen action
            dist_a = dist[range(len(a)), a] 

            with torch.no_grad():
                next_dist = self.g_target[i](z_next)

                next_q = next_dist.mean(dim=-1)  
                next_a = next_q.argmax(dim=1)

                next_dist_a = next_dist[range(len(a)), next_a]  

                target = r.unsqueeze(1) + self.gamma * (1 - done.unsqueeze(1)) * next_dist_a

            loss = quantile_loss(dist_a, target, self.taus)
            total_loss += loss
            losses.append(loss.item())

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # target sync
        if self.total_steps % self.target_update_freq == 0:
            self.f_target.load_state_dict(self.f.state_dict())
            for i in range(self.heads):
                self.g_target[i].load_state_dict(self.g[i].state_dict())

        self.total_steps += 1

        return float(np.mean(losses))
    
    def train(self, num_frames: int, eval_interval: int = 1000, csv_path: str = "results.csv"):

        state, _ = self.envs.reset()

        ep_reward = np.zeros(self.K)
        recent_rewards = []

        for frame in range(1, num_frames + 1):
            actions = self.predict_action(state) 

            next_state, reward, done, truncated, _ = self.envs.step(actions)

            for k in range(self.K):
                self.buffer.add(
                    state[k],
                    actions[k],
                    reward[k],
                    next_state[k],
                    done[k] or truncated[k],
                    {}
                )

            state = next_state
            ep_reward += reward

            for k in range(self.K):
                if done[k] or truncated[k]:
                    recent_rewards.append(ep_reward[k])
                    ep_reward[k] = 0.0

            if len(self.buffer) >= self.batch_size:
                batch = self.buffer.sample(self.batch_size)
                self.update_agent(batch)

            if frame % eval_interval == 0:
                avg = np.mean(recent_rewards[-10:]) if recent_rewards else 0.0
                std = np.std(recent_rewards[-10:]) if recent_rewards else 0.0

                print(
                    f"[Train] Frame {frame}, AvgReward(10): {avg:.2f}"
                )

                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([frame, avg, std])

        print("Training complete.")


@hydra.main(config_path="../configs/agent/", config_name="dqn_ensemble", version_base="1.1")
def main(cfg: DictConfig):
    # build env
    env_vector = [lambda: gym.make(cfg.env.name) for _ in range(cfg.agent.K)]
    envs = gym.vector.SyncVectorEnv(env_vector)
    set_seed(envs, cfg.seed)

    # instantiate & train the agent
    agent = DQNEnsembleAgent(
        envs,
        buffer_capacity=cfg.agent.buffer_capacity,
        batch_size=cfg.agent.batch_size,
        lr=cfg.agent.lr,
        gamma=cfg.agent.gamma,
        target_update_freq=cfg.agent.target_update_freq,
        seed=cfg.seed,
        heads=cfg.agent.heads,
        K=cfg.agent.K,
        phi=cfg.agent.phi,
        alpha=cfg.agent.alpha,
        lmbda=cfg.agent.lmbda,
        N=cfg.agent.N
    )

    run_dir = HydraConfig.get().runtime.output_dir
    csv_dir = os.path.join(run_dir, "dqn_ensemble")
    os.makedirs(csv_dir, exist_ok=True)

    csv_path = os.path.join(csv_dir, f"seed_{cfg.seed}.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "mean_return", "std_return"])
        

    agent.train(cfg.train.num_frames, cfg.train.eval_interval, csv_path=csv_path)


if __name__ == "__main__":
    main()
