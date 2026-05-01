# visualization/visualize/methods/plot_training_curve.py

"""Plot training curve"""

import numpy as np
import matplotlib.pyplot as plt


def plot_training_curve(history, save_path='training_reward.png'):
    """Plot the training reward curve."""
    if not history or 'total_reward' not in history:
        print("No training history to plot")
        return

    rewards = history.get('total_reward', [])
    if not rewards:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    window = min(50, len(rewards)//10)
    if window > 1:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, label='Moving Avg (50)', linewidth=2)

    ax.scatter(range(len(rewards)), rewards, alpha=0.3, s=5, label='Episode Reward')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved to {save_path}")
