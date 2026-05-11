"""
Deep Q-Learning implementation.
"""

from typing import Any, Dict, List, Tuple, Literal

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from rl_exercises.agent import AbstractAgent
from rl_exercises.week_4.buffers import ReplayBuffer
from rl_exercises.week_4.networks import QNetwork


def set_seed(env: gym.Env, seed: int = 0) -> None:
    """
    Seed Python, NumPy, PyTorch and the Gym environment for reproducibility.

    Parameters
    ----------
    env : gym.Env
        The Gym environment to seed.
    seed : int
        Random seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    # some spaces also support .seed()
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)


class DQNAgent(AbstractAgent):
    """
    Deep Q‐Learning agent with ε‐greedy policy and target network.

    Derives from AbstractAgent by implementing:
      - predict_action
      - save / load
      - update_agent
    """

    def __init__(
        self,
        env: gym.Env,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.01,
        epsilon_decay: int = 500,
        target_update_freq: int = 1000,
        num_hidden_layers: int = 2,
        seed: int = 0,
        hidden_dim: int = 64,

        buffer_type: Literal["default", "prioritized"] = "default",
        dqn_type: Literal["default", "double_dqn"] = "default"
    ) -> None:
        """
        Initialize replay buffer, Q‐networks, optimizer, and hyperparameters.

        Parameters
        ----------
        env : gym.Env
            The Gym environment.
        buffer_capacity : int
            Max experiences stored.
        batch_size : int
            Mini‐batch size for updates.
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
            How many updates between target‐network syncs.
        seed : int
            RNG seed.
        
        buffer_type: Literal["default", "prioritized"], optional
            default or prioritized, "default" by default
        dqn_type: Literal["default", "double_dqn"] optional
            default or double_dqn, "default" by default
        """

        assert buffer_type in [
            "default",
            "prioritized",
        ], "buffer_type must be 'default' or 'prioritized'"

        assert dqn_type in [
            "default",
            "double_dqn",
        ], "dqn_type must be 'default' or 'prioritized'"

        super().__init__(
            env,
            buffer_capacity,
            batch_size,
            lr,
            gamma,
            epsilon_start,
            epsilon_final,
            epsilon_decay,
            target_update_freq,
            seed,
        )
        self.env = env
        set_seed(env, seed)

        obs_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n

        # main Q‐network and frozen target
        self.q = QNetwork(obs_dim, n_actions)
        self.q = QNetwork(
            obs_dim,
            n_actions,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
        )

        self.target_q = QNetwork(
            obs_dim,
            n_actions,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
        )

        self.optimizer = optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity, buffer_type = buffer_type)

        # hyperparams
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq

        self.total_steps = 0  # for ε decay and target sync

        self.buffer_type = buffer_type
        self.dqn_type = dqn_type

    def epsilon(self) -> float:
        """
        Compute current ε by exponential decay.

        Returns
        -------
        float
            Exploration rate.
        """
        # TODO: implement exponential‐decayin
        # ε = ε_final + (ε_start - ε_final) * exp(-total_steps / ε_decay)
        # Currently, it is constant and returns the starting value ε
        return self.epsilon_final + (self.epsilon_start - self.epsilon_final) * np.exp(
            -self.total_steps / self.epsilon_decay
        )

    def predict_action(
        self, state: np.ndarray, info: Dict[str, Any] = {}, evaluate: bool = False
    ) -> Tuple[int, Dict]:
        """
        Choose action via ε‐greedy (or purely greedy in eval mode).

        Parameters
        ----------
        state : np.ndarray
            Current observation.
        info : dict
            Gym info dict (unused here).
        evaluate : bool
            If True, always pick argmax(Q).

        Returns
        -------
        action : int
        info_out : dict
            Empty dict (compatible with interface).
        """
        if evaluate:
            # TODO: select purely greedy action from Q(s)
            # purely greedy
            t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                qvals = self.q(t)
            action = int(qvals.argmax(dim=1).item())
        else:
            # ε-greedy
            if np.random.rand() < self.epsilon():
                # TODO: sample random action
                action = self.env.action_space.sample()
            else:
                # TODO: select purely greedy action from Q(s)
                t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    qvals = self.q(t)
                action = int(qvals.argmax(dim=1).item())

        return action

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
                "parameters": self.q.state_dict(),
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
        self.q.load_state_dict(checkpoint["parameters"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def update_agent(
        self, 
        training_batch: List[Tuple[Any, Any, float, Any, bool, Dict]],
        idxs: np.ndarray | None = None,
        weights: np.ndarray | None = None
    ) -> float:
        """
        Perform one gradient update on a batch of transitions.

        Parameters
        ----------
        training_batch : list of transitions
            Each is (state, action, reward, next_state, done, info).
        idxs: np.ndarray | None, optional
            Only relevant for prioritized buffer, None by default
        weights: np.ndarray | None, optional
            Only relevant for prioritized buffer, None by default

        Returns
        -------
        loss_val : float
            MSE loss value.
        """
        # unpack
        states, actions, rewards, next_states, dones, _ = zip(*training_batch)
        s = torch.tensor(np.array(states), dtype=torch.float32)
        a = torch.tensor(np.array(actions), dtype=torch.int64).unsqueeze(1)
        r = torch.tensor(np.array(rewards), dtype=torch.float32)
        s_next = torch.tensor(np.array(next_states), dtype=torch.float32)
        mask = torch.tensor(np.array(dones), dtype=torch.float32)
        if self.buffer_type == "prioritized":
            w = torch.tensor(weights, dtype=torch.float32)

        # current Q estimates for taken actions
        # TODO: pass batched states through self.q and gather Q(s,a)
        pred = self.q(s).gather(1, a).squeeze(1)

        # TODO: compute TD target with frozen network
        with torch.no_grad():
            if self.dqn_type == "double_dqn":
                next_actions = self.q(s_next).argmax(dim=1, keepdim=True)
                next_q = self.target_q(s_next).gather(1, next_actions).squeeze(1)

                target = r + (1 - mask) * self.gamma * next_q
            else:
                next_q = self.target_q(s_next).max(dim=1).values

                target = r + (1 - mask) * self.gamma * next_q
        
        td_errors = target - pred

        loss = (w * (td_errors ** 2)).mean()

        # gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.buffer_type == "prioritized":
            self.buffer.update_priorities(idxs, td_errors.detach().cpu().numpy())

        # occasionally sync target network
        if self.total_steps % self.target_update_freq == 0:
            self.target_q.load_state_dict(self.q.state_dict())

        self.total_steps += 1
        return float(loss.item())

    def train(self, num_frames: int, eval_interval: int = 1000) -> None:
        """
        Run a training loop for a fixed number of frames.

        Parameters
        ----------
        num_frames : int
            Total environment steps.
        eval_interval : int
            Every this many episodes, print average reward.
        """
        state, _ = self.env.reset()
        ep_reward = 0.0
        recent_rewards: List[float] = []

        for frame in range(1, num_frames + 1):
            action = self.predict_action(state)
            next_state, reward, done, truncated, _ = self.env.step(action)

            # store and step
            self.buffer.add(state, action, reward, next_state, done or truncated, {})
            state = next_state
            ep_reward += reward

            # update if ready
            if len(self.buffer) >= self.batch_size:
                # TODO: sample batch from replay buffer
                batch, idxs, weights = self.buffer.sample(batch_size=self.batch_size)
                self.update_agent(batch, idxs, weights)

            if done or truncated:
                state, _ = self.env.reset()
                recent_rewards.append(ep_reward)
                ep_reward = 0.0
                # logging
                if len(recent_rewards) % 10 == 0:
                    # TODO: compute avg over last eval_interval episodes and print
                    avg = np.mean(recent_rewards[-eval_interval:])
                    print(
                        f"Frame {frame}, AvgReward(10): {avg:.2f}, ε={self.epsilon():.3f}"
                    )

        print("Training complete.")

    def train_and_collect_rewards(
        self, num_frames: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train and return episode-end frames and rolling episode rewards.
        """
        state, _ = self.env.reset()
        ep_reward = 0.0

        episode_rewards: List[float] = []
        episode_frames: List[int] = []
        rolling_rewards: List[float] = []

        for frame in range(1, num_frames + 1):
            action = self.predict_action(state)
            next_state, reward, done, truncated, _ = self.env.step(action)

            self.buffer.add(state, action, reward, next_state, done or truncated, {})
            state = next_state
            ep_reward += reward

            if len(self.buffer) >= self.batch_size:
                batch, idxs, weights = self.buffer.sample(batch_size=self.batch_size)
                self.update_agent(batch, idxs, weights)

            if done or truncated:
                state, _ = self.env.reset()

                episode_rewards.append(ep_reward)
                episode_frames.append(frame)
                rolling_rewards.append(np.mean(episode_rewards[-10:]))

                ep_reward = 0.0

        return np.array(episode_frames), np.array(rolling_rewards)


@hydra.main(config_path="../configs/agent/", config_name="dqn", version_base="1.1")
def main(cfg: DictConfig):
    # 1) build env
    env = gym.make(cfg.env.name)
    set_seed(env, cfg.seed)

    # 2) TODO: map config → agent kwargs
    agent_kwargs = dict(
        env=env,
        **cfg.agent,
        seed=cfg.seed,
    )

    # 3) TODO:instantiate & train
    agent = DQNAgent(**agent_kwargs)
    agent.train(cfg.train.num_frames, cfg.train.eval_interval)


if __name__ == "__main__":
    main()
