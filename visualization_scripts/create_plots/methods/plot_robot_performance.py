# visualization_scripts/create_plots/methods/plot_robot_performance.py

"""Plot robot performance"""

import numpy as np
import matplotlib.pyplot as plt


def plot_robot_performance():
    """Robot performance comparison."""
    robots = list(range(10))
    distances = [np.random.randint(20, 80) for _ in robots]
    tasks_completed = [np.random.randint(5, 15) for _ in robots]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    colors = plt.cm.RdYlGn(np.array(tasks_completed)/max(tasks_completed))
    axes[0].bar(robots, distances, color=colors)
    axes[0].set_xlabel('Robot ID')
    axes[0].set_ylabel('Distance Traveled')
    axes[0].set_title('Distance per Robot')
    axes[0].grid(True, alpha=0.3, axis='y')

    x = np.arange(len(robots))
    width = 0.35
    axes[1].bar(x - width/2, distances, width, label='Distance', color='#3498db')
    axes[1].bar(x + width/2, [t*5 for t in tasks_completed], width, label='Tasks x 5', color='#2ecc71')
    axes[1].set_xlabel('Robot ID')
    axes[1].set_ylabel('Value')
    axes[1].set_title('Distance vs Tasks')
    axes[1].set_xticks(x)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('visualizations/02_robot_performance.png', dpi=150)
    plt.close()
