# src/model/attention_policy.py

"""Graph Attention Network for Multi-Agent RL"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from typing import Optional, Tuple


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

        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.gat_convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.gat_convs.append(
                GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

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


class AttentionPolicy(nn.Module):
    def __init__(
        self,
        gnn_encoder: Optional[GNNEncoder] = None,
        state_dim: int = 6,
        action_dim: int = 5,
        hidden_dim: int = 64,
        num_agents: int = 10
    ):
        super().__init__()

        self.gnn_encoder = gnn_encoder or GNNEncoder(
            node_input_dim=state_dim,
            edge_input_dim=1,
            hidden_dim=hidden_dim
        )

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_agents = num_agents

        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim)
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        robot_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ):
        embeddings, _ = self.gnn_encoder(robot_features, edge_index, edge_attr)

        state_pair = self._create_state_pairs(embeddings)

        action_logits = self.policy_head(state_pair)
        action_probs = F.softmax(action_logits, dim=-1)

        state_value = self.value_head(state_pair)

        if return_embeddings:
            return action_probs, state_value, embeddings

        return action_probs, state_value

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
            dist = torch.distributions.Categorical(action_probs)
            actions = dist.sample()

        return actions, state_value


class CentralizedCritic(nn.Module):
    def __init__(
        self,
        gnn_encoder: Optional[GNNEncoder] = None,
        state_dim: int = 6,
        hidden_dim: int = 64,
        num_agents: int = 10
    ):
        super().__init__()

        self.gnn_encoder = gnn_encoder or GNNEncoder(
            node_input_dim=state_dim,
            edge_input_dim=1,
            hidden_dim=hidden_dim
        )

        self.critic = nn.Sequential(
            nn.Linear(hidden_dim + num_agents * 5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

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
