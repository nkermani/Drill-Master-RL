# src/model/attention_policy/gnn_encoder/__init__.py

"""GNN Encoder module using Graph Attention Networks"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from typing import Optional, Tuple

from .components import NodeEncoder, EdgeEncoder, GATConvs, OutputProj
from .forward import forward as forward_fn


class GNNEncoder(nn.Module):
    def __init__(
        self,
        node_input_dim: int = 6,
        edge_input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.node_encoder = NodeEncoder(node_input_dim, hidden_dim, dropout)
        self.edge_encoder = EdgeEncoder(edge_input_dim, hidden_dim)

        self.gat_convs = GATConvs(hidden_dim, num_layers, num_heads, dropout)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        self.output_proj = OutputProj(hidden_dim)

    forward = forward_fn


__all__ = ['GNNEncoder']
