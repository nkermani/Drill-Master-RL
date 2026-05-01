# src/model/qmix/qmix/methods/update.py

"""Update method for QMIX"""

import torch
import torch.nn.functional as F


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
