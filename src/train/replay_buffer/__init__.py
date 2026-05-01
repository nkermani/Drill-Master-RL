# src/train/replay_buffer/__init__.py

"""Replay Buffer module for training"""

import numpy as np
from typing import Tuple

from .methods import _allocate, add, sample


class ReplayBuffer:
    def __init__(self, capacity: int = 100000, num_agents: int = 10):
        self.capacity = capacity
        self.num_agents = num_agents
        self.position = 0
        self.size = 0

        self.obs = None
        self.actions = None
        self.rewards = None
        self.next_obs = None
        self.dones = None

    _allocate = _allocate
    add = add
    sample = sample


__all__ = ['ReplayBuffer']
