"""QMIX-style Mixing Network for Multi-Agent RL"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MixingNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int = 5,
        num_agents: int = 10,
        hidden_dim: int = 64
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_agents * hidden_dim)
        )
        
        self.hyper_b1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(
        self,
        agent_qs: torch.Tensor,
        state: torch.Tensor
    ) -> torch.Tensor:
        w1 = torch.abs(self.hyper_w1(state))
        w1 = w1.reshape(-1, self.num_agents, self.hidden_dim)
        
        b1 = self.hyper_b1(state).reshape(-1, 1, self.hidden_dim)
        
        hidden = agent_qs.unsqueeze(-1)
        
        hidden = torch.bmm(hidden, w1) + b1
        hidden = F.relu(hidden)
        
        w2 = torch.abs(self.hyper_w2(state))
        w2 = w2.reshape(-1, self.hidden_dim, 1)
        
        b2 = self.hyper_b2(state).reshape(-1, 1, 1)
        
        q_tot = torch.bmm(hidden, w2) + b2
        q_tot = q_tot.squeeze(-1)
        
        return q_tot


class QMIX:
    def __init__(
        self,
        agents: nn.ModuleList,
        mixing_network: MixingNetwork,
        optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        target_update_interval: int = 200
    ):
        self.agents = agents
        self.mixing_network = mixing_network
        self.optimizer = optimizer
        self.gamma = gamma
        self.target_update_interval = target_update_interval
        self.update_count = 0
        
        self.target_agents = [agent.clone() for agent in agents]
        self.target_mixing_network = MixingNetwork(
            state_dim=5,
            num_agents=10,
            hidden_dim=64
        )
        self.target_mixing_network.load_state_dict(mixing_network.state_dict())
    
    def update(
        self,
        experiences: list,
        state_batch: torch.Tensor
    ) -> dict:
        obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch = experiences
        
        agent_qs = []
        target_agent_qs = []
        
        for i, agent in enumerate(self.agents):
            agent_q = agent(
                obs_batch[:, i],
                torch.zeros(10, 2, dtype=torch.long),
                torch.ones(10, 1, dtype=torch.float32)
            )[:, 0]
            agent_qs.append(agent_q)
        
        agent_qs = torch.stack(agent_qs, dim=1)
        
        for i, agent in enumerate(self.target_agents):
            target_q = agent(
                next_obs_batch[:, i],
                torch.zeros(10, 2, dtype=torch.long),
                torch.ones(10, 1, dtype=torch.float32)
            )[:, 0]
            target_agent_qs.append(target_q)
        
        target_agent_qs = torch.stack(target_agent_qs, dim=1)
        
        target_q_max = target_agent_qs.max(dim=2, keepdim=True)[0]
        
        rewards = rewards_batch.sum(dim=1, keepdim=True)
        dones = dones_batch.any(dim=1, keepdim=True)
        
        target_q_tot = rewards + self.gamma * (1 - dones.float()) * target_q_max
        
        q_tot = self.mixing_network(agent_qs, state_batch)
        
        loss = F.mse_loss(q_tot, target_q_detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agents.parameters(), 10.0)
        self.optimizer.step()
        
        self.update_count += 1
        
        if self.update_count % self.target_update_interval == 0:
            self._update_target_networks()
        
        return {'loss': loss.item()}
    
    def _update_target_networks(self):
        for target_agent, agent in zip(self.target_agents, self.agents):
            target_agent.load_state_dict(agent.state_dict())
        
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())


def target_q_detach():
    return torch.detach


class ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, *experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int):
        indices = torch.randint(0, len(self.buffer), (batch_size,))
        
        obs, actions, rewards, next_obs, dones = [], [], [], [], []
        
        for idx in indices:
            o, a, r, no, d = self.buffer[idx]
            obs.append(o)
            actions.append(a)
            rewards.append(r)
            next_obs.append(no)
            dones.append(d)
        
        return (
            torch.stack(obs),
            torch.stack(actions),
            torch.stack(rewards),
            torch.stack(next_obs),
            torch.stack(dones)
        )
    
    def __len__(self):
        return len(self.buffer)