# src/model/attention_policy/centralized_critic/__init__.py

"""Centralized Critic module for Multi-Agent RL"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ..gnn_encoder import GNNEncoder
from .components import Critic
from .forward import forward as forward_fn


class CentralizedCritic(nn.Module):
    def __init__(
        self,
        gnn_encoder: Optional[GNNEncoder] = None,
        state_dim: int = 6,
        hidden_dim: int = 64,
        num_agents: int = 10
    ):
        super().__init__()

        self.gnn_encoder = gnn_encoder or GNNEncoder(
            node_input_dim=state_dim,
            edge_input_dim=1,
            hidden_dim=hidden_dim
        )

        self.critic = Critic(hidden_dim, num_agents)

    forward = forward_fn


__all__ = ['CentralizedCritic']
