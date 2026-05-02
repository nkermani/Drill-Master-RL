# src/model/qmix/mixing_network/components/hyper_b2.py

"""Hyper network b2 component"""

import torch
import torch.nn as nn


class HyperB2(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
