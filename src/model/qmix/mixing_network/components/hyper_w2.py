# src/model/qmix/mixing_network/components/hyper_w2.py

"""Hyper network w2 component"""

import torch.nn as nn


class HyperW2(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
