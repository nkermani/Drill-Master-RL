# src/model/attention_policy/attention_policy/methods/create_state_pairs.py

"""Create state pairs method for AttentionPolicy"""

import torch


def _create_state_pairs(self, embeddings: torch.Tensor) -> torch.Tensor:
    num_agents = embeddings.shape[0]

    agent_idx = torch.arange(num_agents, device=embeddings.device)
    agent_idx_1 = agent_idx.unsqueeze(1).expand(num_agents, num_agents)
    agent_idx_2 = agent_idx.unsqueeze(0).expand(num_agents, num_agents)

    emb1 = embeddings[agent_idx_1.reshape(-1)]
    emb2 = embeddings[agent_idx_2.reshape(-1)]

    state_pairs = torch.cat([emb1, emb2], dim=-1).reshape(num_agents, num_agents, -1)

    max_pairs = state_pairs.size(1)
    state_pairs = state_pairs[:, :max_pairs, :].mean(dim=1)

    return state_pairs
