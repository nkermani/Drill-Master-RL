# src/model/attention_policy/gnn_encoder/forward.py

"""Forward method for GNNEncoder"""

import torch.nn.functional as F
from typing import Optional, Tuple


def forward(
    self,
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor] = None,
    batch: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = self.node_encoder(node_features)

    for i in range(self.num_layers):
        x_new = self.gat_convs[i](x, edge_index)
        x_new = self.norms[i](x_new)
        x_new = F.relu(x_new)
        x = x_new + x

    embeddings = self.output_proj(x)

    return embeddings, edge_index
