# src/train/ppo_agent/methods/update.py

"""Update method for PPOAgent"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


def update(
    self,
    batch: tuple,
    epoch: int
) -> Dict:
    obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch = batch

    action_probs, values = self.policy(obs_batch, torch.zeros(obs_batch.shape[1], 2, dtype=torch.long))

    returns = rewards_batch.view(-1)

    advantages = returns - values.detach()

    dist = torch.distributions.Categorical(action_probs)
    log_probs = dist.log_prob(actions_batch.view(-1))

    entropy = dist.entropy().mean()

    policy_loss = -(log_probs * advantages.detach()).mean()

    value_loss = F.mse_loss(values, returns.detach())

    loss = (
        policy_loss
        + self.value_coef * value_loss
        - self.entropy_coef * entropy
    )

    self.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
    self.optimizer.step()

    return {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'entropy': entropy.item(),
        'total_loss': loss.item()
    }
