# src/train/trainer/methods/collect_experience.py

"""Collect experience method for Trainer"""

import torch
from typing import Dict


def collect_experience(self, num_steps: int = 100) -> Dict:
    obs, info = self.env.reset()

    total_reward = 0
    total_steps = 0

    for _ in range(num_steps):
        robot_features = torch.tensor(obs['robot_features'], dtype=torch.float32)

        if robot_features.dim() == 3:
            robot_features = robot_features.squeeze(0)

        edge_index = torch.tensor([[0], [0]], dtype=torch.long)

        with torch.no_grad():
            action_probs, value = self.policy(robot_features, edge_index)

        actions = action_probs[0].argmax(dim=-1).numpy()
        if actions.ndim == 0:
            actions = [int(actions)]
        else:
            actions = actions.tolist()

        next_obs, rewards, terminations, truncations, info = self.env.step(actions)

        self.replay_buffer.add(
            obs['robot_features'],
            actions,
            rewards,
            next_obs['robot_features'],
            terminations
        )

        total_reward += sum(rewards)
        total_steps += 1

        obs = next_obs

        if any(truncations):
            break

    return {
        'total_reward': total_reward,
        'total_steps': total_steps,
        'info': info
    }
