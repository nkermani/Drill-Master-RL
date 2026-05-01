# src/model/attention_policy/attention_policy/methods/get_action.py

"""Get action method for AttentionPolicy"""

import torch
from torch.distributions import Categorical
from typing import Tuple


def get_action(
    self,
    robot_features: torch.Tensor,
    edge_index: torch.Tensor,
    deterministic: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    action_probs, state_value = self.forward(robot_features, edge_index)

    if deterministic:
        actions = action_probs.argmax(dim=-1)
    else:
        dist = Categorical(action_probs)
        actions = dist.sample()

    return actions, state_value
