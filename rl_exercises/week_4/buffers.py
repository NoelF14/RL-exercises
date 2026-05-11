from typing import Any, Dict, List, Tuple, Literal

import random

import numpy as np
from rl_exercises.agent import AbstractBuffer


class ReplayBuffer(AbstractBuffer):
    """
    Simple FIFO replay buffer.

    Stores tuples of (state, action, reward, next_state, done, info),
    and evicts the oldest when capacity is exceeded.

    Addition Level 3: Prioritized Experience Replay Buffer
    """

    def __init__(
        self, 
        capacity: int,
        buffer_type: Literal["default", "prioritized"] = "default",
        alpha: float = 0.6,   
        beta: float = 0.4, 
        beta_increment: float = 1e-3
    ) -> None:
        """
        Parameters
        ----------
        capacity : int
            Maximum number of transitions to store.
        buffer_type: Literal["default", "prioritized"], optional
            default or prioritized, "default" by default
        alpha: float, optional
            how much prioritization isused, 0.6 by default
        beta: float, optional
            used to calculate weights for correcting the bias, 0.4 by default
        beta_increment, optional
            anneal beta from its initial value to 1, 1e-3 by default

        
        """

        assert buffer_type in [
            "default",
            "prioritized",
        ], "type must be 'default' or 'prioritized'"

        super().__init__()
        self.capacity = capacity
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.next_states: List[np.ndarray] = []
        self.dones: List[bool] = []
        self.infos: List[Dict] = []

        self.buffer_type = buffer_type
        if (self.buffer_type == "prioritized"):
            self.alpha = alpha
            self.beta = beta
            self.beta_increment = beta_increment
            self.priorities: List[float] = []


    def add(
        self,
        state: np.ndarray,
        action: int | float,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: dict,
    ) -> None:
        """
        Add a single transition to the buffer.

        If the buffer is full, the oldest transition is removed.

        Parameters
        ----------
        state : np.ndarray
            Observation before action.
        action : int or float
            Action taken.
        reward : float
            Reward received.
        next_state : np.ndarray
            Observation after action.
        done : bool
            Whether episode terminated/truncated.
        info : dict
            Gym info dict (can store extras).
        """

        if self.buffer_type == "prioritized":
            max_priority = max(self.priorities, default=1.0)

        if len(self.states) >= self.capacity:
            # TODO: pop the oldest element off each list (states, actions, …, infos)
            # pop oldest
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
            self.dones.pop(0)
            self.infos.pop(0)
            if self.buffer_type == "prioritized":
                self.priorities.pop(0)

        # TODO: append state, action, reward, next_state, done, info to their respective lists
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.infos.append(info)

        if self.buffer_type == "prioritized":
            self.priorities.append(max_priority)

        return

    def sample(
        self, batch_size: int = 32
    ) -> Tuple[
        List[Tuple[Any, Any, float, Any, bool, Dict]],
        np.ndarray | None,
        np.ndarray | None
    ]:
        """
        Uniformly sample a batch of transitions.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample.

        Returns
        -------
        List of transitions as (state, action, reward, next_state, done, info).
        Indexes (only relevant for prioritized)
        Weights (only relevant for prioritized)
        """
        if self.buffer_type == "prioritized":
            priorities = np.array(self.priorities, dtype=np.float32)
            scaled_priorities = priorities ** self.alpha

            probs = scaled_priorities / np.sum(scaled_priorities)

            N = len(self.priorities)
            idxs = np.random.choice(N, batch_size, p=probs)
            weights = (N * probs[idxs]) ** (-self.beta)
            weights /= weights.max()  # normalize

            self.beta = min(1.0, self.beta + self.beta_increment)
        else:
            idxs = random.sample(range(len(self.states)), batch_size) 

        batch = [
            (
                self.states[i],
                self.actions[i],
                self.rewards[i],
                self.next_states[i],
                self.dones[i],
                self.infos[i],
            )
            for i in idxs
        ]

        if self.buffer_type == "prioritized":
            return batch, idxs, weights
        else:
            return batch, None, None
    
    def update_priorities(self, idxs: List[int], td_errors: np.ndarray):
        """
        Update priorities p_i <- |δ_i|
        """
        for i, err in zip(idxs, td_errors):
            self.priorities[i] = abs(err) + 1e-6  # avoid zero priority

    def __len__(self) -> int:
        """Current number of stored transitions."""
        return len(self.states)
