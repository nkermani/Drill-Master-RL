# src/train/trainer/methods/load_checkpoint.py

"""Load checkpoint method for Trainer"""

import torch


def load_checkpoint(self, path: str):
    checkpoint = torch.load(path)

    self.policy.load_state_dict(checkpoint['policy_state_dict'])
    self.value_network.load_state_dict(checkpoint['value_state_dict'])
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
