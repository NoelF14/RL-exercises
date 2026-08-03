"""Reproducibility controls shared by training and evaluation."""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from stable_baselines3.common.utils import set_random_seed


def seed_everything(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy, PyTorch, and Stable-Baselines3."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_random_seed(seed, using_cuda=torch.cuda.is_available())
    torch.use_deterministic_algorithms(deterministic_torch, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic_torch
