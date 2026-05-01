# src/train/ppo_agent/__init__.py

"""PPO Agent module"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from .methods import compute_returns, update


class PPOAgent:
    def __init__(
        self,
        policy,
        value_network,
        optimizer,
        num_agents: int = 10,
        action_dim: int = 5,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5
    ):
        self.policy = policy
        self.value_network = value_network
        self.optimizer = optimizer

        self.num_agents = num_agents
        self.action_dim = action_dim
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

    compute_returns = compute_returns
    update = update


__all__ = ['PPOAgent']
