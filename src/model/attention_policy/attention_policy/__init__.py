# src/model/attention_policy/attention_policy/__init__.py

"""Attention Policy module for Multi-Agent RL"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from ..gnn_encoder import GNNEncoder
from .components import PolicyHead, ValueHead
from .methods import forward, create_state_pairs, get_action


class AttentionPolicy(nn.Module):
    def __init__(
        self,
        gnn_encoder: Optional[GNNEncoder] = None,
        state_dim: int = 6,
        action_dim: int = 5,
        hidden_dim: int = 64,
        num_agents: int = 10
    ):
        super().__init__()

        self.gnn_encoder = gnn_encoder or GNNEncoder(
            node_input_dim=state_dim,
            edge_input_dim=1,
            hidden_dim=hidden_dim
        )

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_agents = num_agents

        self.policy_head = PolicyHead(hidden_dim, action_dim)
        self.value_head = ValueHead(hidden_dim)

    forward = forward
    _create_state_pairs = create_state_pairs
    get_action = get_action


__all__ = ['AttentionPolicy']
