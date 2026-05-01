# src/model/attention_policy/gnn_encoder/components/gat_convs.py

"""GAT convolutions component"""

import torch.nn as nn
from torch_geometric.nn import GATConv


class GATConvs(nn.ModuleList):
    def __init__(self, hidden_dim: int, num_layers: int, num_heads: int, dropout: float):
        super().__init__()
        for _ in range(num_layers):
            self.append(
                GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
            )
