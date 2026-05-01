# src/model/attention_policy/attention_policy/components/policy_head.py

"""Policy head component"""

import torch.nn as nn


class PolicyHead(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
