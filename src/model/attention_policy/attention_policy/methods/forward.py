# src/model/attention_policy/attention_policy/methods/forward.py

"""Forward method for AttentionPolicy"""

import torch
import torch.nn.functional as F
from typing import Optional


def forward(
    self,
    robot_features: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor] = None,
    return_embeddings: bool = False
):
    if robot_features.dim() == 3:
        robot_features = robot_features.squeeze(0)

    embeddings, _ = self.gnn_encoder(robot_features, edge_index, edge_attr)

    state_pair = self._create_state_pairs(embeddings)

    action_logits = self.policy_head(state_pair)
    action_probs = F.softmax(action_logits, dim=-1)

    state_value = self.value_head(state_pair).squeeze(-1)

    if return_embeddings:
        return action_probs, state_value, embeddings

    return action_probs, state_value
