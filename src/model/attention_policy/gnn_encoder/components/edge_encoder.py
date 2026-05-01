# src/model/attention_policy/gnn_encoder/components/edge_encoder.py

"""Edge encoder component"""

import torch.nn as nn


class EdgeEncoder(nn.Module):
    def __init__(self, edge_input_dim: int, hidden_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
