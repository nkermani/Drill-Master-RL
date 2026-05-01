# src/train/trainer/methods/save_checkpoint.py

"""Save checkpoint method for Trainer"""

import os
import torch


def save_checkpoint(self, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        'policy_state_dict': self.policy.state_dict(),
        'value_state_dict': self.value_network.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
    }

    torch.save(checkpoint, path)
