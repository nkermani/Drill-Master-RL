# src/model/qmix/replay_buffer/methods/sample.py

"""Sample method for ReplayBuffer"""

import torch


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
