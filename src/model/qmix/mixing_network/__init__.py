# src/model/qmix/mixing_network/__init__.py

"""Mixing Network module for QMIX"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .components import HyperW1, HyperB1, HyperW2, HyperB2
from .forward import forward as forward_fn


class MixingNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int = 5,
        num_agents: int = 10,
        hidden_dim: int = 64
    ):
        super().__init__()

        self.state_dim = state_dim
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim

        self.hyper_w1 = HyperW1(state_dim, num_agents, hidden_dim)
        self.hyper_b1 = HyperB1(state_dim, hidden_dim)
        self.hyper_w2 = HyperW2(state_dim, hidden_dim)
        self.hyper_b2 = HyperB2(state_dim, hidden_dim)

    forward = forward_fn


__all__ = ['MixingNetwork']
