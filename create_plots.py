"""Generate sample visualizations for N-Drill-Master-RL"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os


def create_sample_plots():
    """Create sample visualization plots."""
    os.makedirs('visualizations', exist_ok=True)
    
    plot_training_progress()
    plot_robot_performance()
    plot_warehouse_layout()
    plot_gnn_attention()
    
    print("Created visualization files in visualizations/")


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


def plot_warehouse_layout():
    """Sample warehouse grid layout."""
    grid_size = (10, 10)
    num_robots = 8
    num_stations = 6
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            rect = patches.Rectangle((y-0.45, grid_size[0]-x-0.55), 0.9, 0.9,
                                   fill=False, edgecolor='#ddd', linewidth=1)
            ax.add_patch(rect)
    
    station_locs = [(1, 1), (1, 8), (8, 1), (8, 8), (1, 4), (8, 4)]
    for sx, sy in station_locs[:num_stations]:
        rect = patches.Rectangle((sy-0.4, grid_size[0]-sx-0.6), 0.8, 0.8,
                               linewidth=2, edgecolor='green', facecolor='lightgreen', alpha=0.7)
        ax.add_patch(rect)
    
    robot_locs = [(3, 2), (3, 5), (4, 4), (5, 6), (6, 3), (2, 7), (7, 2), (4, 7)]
    colors = plt.cm.tab10(np.linspace(0, 1, num_robots))
    for i, (rx, ry) in enumerate(robot_locs[:num_robots]):
        circle = plt.Circle((ry, grid_size[0]-rx-1), 0.35, color=colors[i], alpha=0.8)
        ax.add_patch(circle)
        ax.text(ry, grid_size[0]-rx-1, str(i), ha='center', va='center', 
               fontsize=9, fontweight='bold', color='white')
    
    task_lines = [(3, 5, 8, 8), (5, 6, 1, 1), (6, 3, 8, 4)]
    for sx, sy, dx, dy in task_lines:
        ax.annotate('', xy=(grid_size[0]-dx-1, dy), xytext=(grid_size[0]-sx-1, sy),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    ax.set_xlim(-0.5, grid_size[1]-0.5)
    ax.set_ylim(-0.5, grid_size[0]-0.5)
    ax.set_xticks(range(grid_size[1]))
    ax.set_yticks(range(grid_size[0]))
    ax.set_xticklabels(range(grid_size[1]))
    ax.set_yticklabels(list(range(grid_size[0]-1, -1, -1)))
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_title('Warehouse Layout (Stations: green, Robots: colored, Tasks: red arrows)')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/03_warehouse_layout.png', dpi=150)
    plt.close()


def plot_gnn_attention():
    """Sample GNN attention heatmap."""
    np.random.seed(42)
    num_agents = 10
    
    attention = np.random.rand(num_agents, num_agents)
    attention = (attention + attention.T) / 2
    np.fill_diagonal(attention, 0)
    attention = attention / attention.sum(axis=1, keepdims=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    im = axes[0].imshow(attention, cmap='YlOrRd', aspect='auto')
    axes[0].set_xlabel('To Robot')
    axes[0].set_ylabel('From Robot')
    axes[0].set_title('GNN Attention Weights')
    axes[0].set_xticks(range(num_agents))
    axes[0].set_yticks(range(num_agents))
    plt.colorbar(im, ax=axes[0])
    
    layers = ['Input', 'GAT-1', 'GAT-2', 'GAT-3', 'Policy']
    values = [6, 64, 64, 64, 5]
    axes[1].bar(layers, values, color=['#3498db', '#2ecc71', '#2ecc71', '#2ecc71', '#e74c3c'])
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Hidden Dim')
    axes[1].set_title('Network Architecture')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(values):
        axes[1].text(i, v + 2, str(v), ha='center')
    
    plt.tight_layout()
    plt.savefig('visualizations/04_gnn_attention.png', dpi=150)
    plt.close()


if __name__ == '__main__':
    create_sample_plots()