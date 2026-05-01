# src/train/trainer/methods/train.py

"""Train method for Trainer"""

import torch
from typing import Dict
from tqdm import tqdm


def train(self, num_updates: int = 1000, log_interval: int = 10) -> Dict:
    history = {
        'policy_loss': [],
        'value_loss': [],
        'entropy': [],
        'total_reward': []
    }

    for update in tqdm(range(num_updates), desc="Training"):
        exp_info = self.collect_experience(num_steps=self.max_steps)

        for _ in range(self.num_epochs):
            batch = self.replay_buffer.sample(self.batch_size)

            if batch is None:
                break

            update_stats = self.agent.update(batch, epoch=update)

            history['policy_loss'].append(update_stats['policy_loss'])
            history['value_loss'].append(update_stats['value_loss'])
            history['entropy'].append(update_stats['entropy'])

        history['total_reward'].append(exp_info['total_reward'])

        if update % log_interval == 0:
            print(f"Update {update}: Reward={exp_info['total_reward']:.2f}")

    return history
