# src/model/attention_policy/attention_policy/methods/create_state_pairs.py

"""Create state pairs method for AttentionPolicy"""

import torch


def _create_state_pairs(self, embeddings: torch.Tensor) -> torch.Tensor:
    if embeddings.dim() == 3:
        batch_size, num_agents, hidden_dim = embeddings.shape
        embeddings = embeddings.reshape(-1, hidden_dim)
    
    if embeddings.dim() == 2:
        num_agents = embeddings.shape[0]
        hidden_dim = embeddings.shape[1]
        
        mean_embedding = embeddings.mean(dim=0, keepdim=True).expand(num_agents, -1)
        state_pairs = torch.cat([embeddings, mean_embedding], dim=-1)
        return state_pairs
    
    return embeddings
