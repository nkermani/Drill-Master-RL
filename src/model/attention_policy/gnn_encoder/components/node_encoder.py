# src/model/attention_policy/gnn_encoder/components/node_encoder.py

"""Node encoder component"""

import torch
import torch.nn as nn


class NodeEncoder(nn.Module):
    def __init__(self, node_input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
