# src/train/ppo_agent/methods/compute_returns.py

"""Compute returns method for PPOAgent"""

import torch


def compute_returns(
    self,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor
) -> tuple:
    returns = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)

    for t in reversed(range(rewards.shape[0])):
        if t == rewards.shape[0] - 1:
            next_val = torch.zeros_like(values[0])
        else:
            next_val = values[t + 1]

        returns[t] = rewards[t] + self.gamma * next_val * (1 - dones[t].float())
        advantages[t] = returns[t] - values[t]

    return returns, advantages
