# src/model/qmix/qmix/__init__.py

"""QMIX Algorithm module"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ..mixing_network import MixingNetwork
from .methods import update, update_target_networks


class QMIX:
    def __init__(
        self,
        agents: nn.ModuleList,
        mixing_network: MixingNetwork,
        optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        target_update_interval: int = 200
    ):
        self.agents = agents
        self.mixing_network = mixing_network
        self.optimizer = optimizer
        self.gamma = gamma
        self.target_update_interval = target_update_interval
        self.update_count = 0

        self.target_agents = [agent.clone() for agent in agents]
        self.target_mixing_network = MixingNetwork(
            state_dim=5,
            num_agents=10,
            hidden_dim=64
        )
        self.target_mixing_network.load_state_dict(mixing_network.state_dict())

    update = update
    _update_target_networks = update_target_networks


__all__ = ['QMIX']
