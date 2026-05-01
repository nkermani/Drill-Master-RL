# src/model/qmix/replay_buffer/__init__.py

"""Replay Buffer module"""

import torch
from typing import Optional

from .methods import push, sample


class ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    push = push
    sample = sample

    def __len__(self):
        return len(self.buffer)


__all__ = ['ReplayBuffer']
