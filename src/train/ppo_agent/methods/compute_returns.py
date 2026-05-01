# src/train/ppo_agent/methods/compute_returns.py

"""Compute returns method for PPOAgent"""


def compute_returns(
    self,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor
) -> tuple:
    advantages = torch.zeros_like(rewards)

    gae = 0
    for t in reversed(range(rewards.shape[0])):
        if t == rewards.shape[0] - 1:
            next_value = 0
        else:
            next_value = next_values[t]

        delta = rewards[t] + self.gamma * next_value * (1 - dones[t].float()) - values[t]
        gae = delta + self.gamma * self.lam * (1 - dones[t].float()) * gae
        advantages[t] = gae

    returns = advantages + values

    return returns, advantages
