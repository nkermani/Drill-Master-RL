# src/train/replay_buffer/methods/_allocate.py

"""Allocate method for ReplayBuffer"""

import numpy as np


def _allocate(self, batch_size: int):
    if self.obs is None:
        max_capacity = self.capacity

        self.obs = np.zeros((max_capacity, self.num_agents, 6), dtype=np.float32)
        self.actions = np.zeros((max_capacity, self.num_agents), dtype=np.int64)
        self.rewards = np.zeros((max_capacity, self.num_agents), dtype=np.float32)
        self.next_obs = np.zeros((max_capacity, self.num_agents, 6), dtype=np.float32)
        self.dones = np.zeros((max_capacity, self.num_agents), dtype=np.bool_)
