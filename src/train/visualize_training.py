"""Visualization function for training history"""

from typing import Dict


def visualize_training(history: Dict, save_path: str = 'training_curves.png'):
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        axes[0, 0].plot(history['policy_loss'])
        axes[0, 0].set_title('Policy Loss')
        axes[0, 0].set_xlabel('Update')
        
        axes[0, 1].plot(history['value_loss'])
        axes[0, 1].set_title('Value Loss')
        axes[0, 1].set_xlabel('Update')
        
        axes[1, 0].plot(history['entropy'])
        axes[1, 0].set_title('Entropy')
        axes[1, 0].set_xlabel('Update')
        
        axes[1, 1].plot(history['total_reward'])
        axes[1, 1].set_title('Total Reward')
        axes[1, 1].set_xlabel('Update')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    except ImportError:
        print("Matplotlib not available for visualization")
