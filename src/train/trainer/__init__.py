# src/train/trainer/__init__.py

"""Trainer module"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional

from ..ppo_agent import PPOAgent
from ..replay_buffer import ReplayBuffer
from .methods import collect_experience, train, save_checkpoint, load_checkpoint


class Trainer:
    def __init__(
        self,
        env,
        policy,
        value_network,
        num_agents: int = 10,
        learning_rate: float = 3e-4,
        buffer_capacity: int = 100000,
        batch_size: int = 64,
        num_epochs: int = 10,
        max_steps: int = 1000
    ):
        self.env = env
        self.policy = policy
        self.value_network = value_network

        params = list(policy.parameters()) + list(value_network.parameters())
        self.optimizer = optim.Adam(params, lr=learning_rate)

        self.agent = PPOAgent(
            policy=policy,
            value_network=value_network,
            optimizer=self.optimizer,
            num_agents=num_agents
        )

        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity, num_agents=num_agents)

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.max_steps = max_steps

    collect_experience = collect_experience
    train = train
    save_checkpoint = save_checkpoint
    load_checkpoint = load_checkpoint


__all__ = ['Trainer']
