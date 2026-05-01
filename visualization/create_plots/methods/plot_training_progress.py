# visualization/create_plots/methods/plot_training_progress.py

"""Plot training progress"""

import numpy as np
import matplotlib.pyplot as plt


def plot_training_progress():
    """Sample training reward curve."""
    np.random.seed(42)
    episodes = 1000
    rewards = []

    base = 50
    for i in range(episodes):
        noise = np.random.randn() * 20
        progress = (1 - np.exp(-i/300)) * 150
        rewards.append(base + progress + noise)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    window = 50
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')

    axes[0].plot(rewards, alpha=0.3, color='blue', label='Episode')
    axes[0].plot(range(window-1, len(rewards)), smoothed, color='blue', linewidth=2, label='Moving Avg')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Training Reward Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    policies = ['PPO', 'QMIX', 'MAPPO', 'IQL']
    final_rewards = [245, 210, 195, 180]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    axes[1].bar(policies, final_rewards, color=colors)
    axes[1].set_xlabel('Algorithm')
    axes[1].set_ylabel('Final Reward')
    axes[1].set_title('Algorithm Comparison')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('visualizations/01_training_progress.png', dpi=150)
    plt.close()
