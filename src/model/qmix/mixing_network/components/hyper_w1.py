# src/model/qmix/mixing_network/components/hyper_w1.py

"""Hyper network w1 component"""

import torch
import torch.nn as nn


class HyperW1(nn.Module):
    def __init__(self, state_dim: int, num_agents: int, hidden_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_agents * hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
