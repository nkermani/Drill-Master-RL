# src/model/attention_policy/attention_policy/components/value_head.py

"""Value head component"""

import torch.nn as nn


class ValueHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
