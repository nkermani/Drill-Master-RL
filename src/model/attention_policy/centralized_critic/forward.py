# src/model/attention_policy/centralized_critic/forward.py

"""Forward method for CentralizedCritic"""

import torch
import torch.nn.functional as F
from typing import Optional


def forward(
    self,
    robot_features: torch.Tensor,
    edge_index: torch.Tensor,
    actions: torch.Tensor,
    edge_attr: Optional[torch.Tensor] = None
) -> torch.Tensor:
    embeddings, _ = self.gnn_encoder(robot_features, edge_index, edge_attr)

    global_embedding = embeddings.mean(dim=0, keepdim=True).expand(embeddings.shape[0], -1)

    action_encoding = F.one_hot(actions, num_classes=5).float()

    combined = torch.cat([global_embedding, action_encoding], dim=-1)

    q_values = self.critic(combined)

    return q_values.squeeze(-1)
