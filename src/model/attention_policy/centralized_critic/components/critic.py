# src/model/attention_policy/centralized_critic/components/critic.py

"""Critic component for CentralizedCritic"""

import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, hidden_dim: int, num_agents: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim + num_agents * 5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
