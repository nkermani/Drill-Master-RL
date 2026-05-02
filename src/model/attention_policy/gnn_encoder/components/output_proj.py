# src/model/attention_policy/gnn_encoder/components/output_proj.py

"""Output projection component"""

import torch
import torch.nn as nn


class OutputProj(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
