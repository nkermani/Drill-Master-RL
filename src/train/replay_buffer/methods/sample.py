# src/train/replay_buffer/methods/sample.py

"""Sample method for ReplayBuffer"""

import torch
import numpy as np


def sample(self, batch_size: int) -> tuple:
    if self.size < batch_size:
        return None

    indices = np.random.randint(0, self.size, size=batch_size)

    return (
        torch.tensor(self.obs[indices], dtype=torch.float32),
        torch.tensor(self.actions[indices], dtype=torch.long),
        torch.tensor(self.rewards[indices], dtype=torch.float32),
        torch.tensor(self.next_obs[indices], dtype=torch.float32),
        torch.tensor(self.dones[indices], dtype=torch.bool)
    )
