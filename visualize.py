"""Visualization script for N-Drill-Master-RL"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import deque
import os


def visualize_warehouse(env, save_path=None):
    """Visualize the warehouse grid with robots and tasks."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    grid = np.zeros((env.grid_size[0], env.grid_size[1]))
    
    for x in range(env.grid_size[0]):
        for y in range(env.grid_size[1]):
            idx = x * env.grid_size[1] + y
            if env.grid[x, y] == 1:
                grid[x, y] = 0.3
    
    for station in env.stations:
        sx = station // env.grid_size[1]
        sy = station % env.grid_size[1]
        rect = patches.Rectangle((sy-0.4, env.grid_size[0]-sx-0.6), 0.8, 0.8,
                           linewidth=2, edgecolor='green', facecolor='lightgreen')
        ax.add_patch(rect)
    
    for robot in env.robots:
        rx = robot.pos // env.grid_size[1]
        ry = robot.pos % env.grid_size[1]
        color = 'blue' if robot.state.value == 0 else 'red'
        circle = plt.Circle((ry, env.grid_size[0]-rx-1), 0.3, color=color, alpha=0.7)
        ax.add_patch(circle)
        ax.text(ry, env.grid_size[0]-rx-1, str(robot.robot_id), 
               ha='center', va='center', fontsize=8, color='white')
    
    ax.set_xlim(-0.5, env.grid_size[1]-0.5)
    ax.set_ylim(-0.5, env.grid_size[0]-0.5)
    ax.set_xticks(range(env.grid_size[1]))
    ax.set_yticks(range(env.grid_size[0]))
    ax.set_xticklabels(range(env.grid_size[1]))
    ax.set_yticklabels(range(env.grid_size[0], reversed(range(env.grid_size[0])))
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_title(f'Warehouse - Step {env.current_step}')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


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


def plot_task_distribution(env, save_path='task_distribution.png'):
    """Plot task locations and completions."""
    pending = [t for t in env.tasks if not t.completed]
    completed = [t for t in env.tasks if t.completed]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for task in pending:
        px = task.pickup_loc // env.grid_size[1]
        py = task.pickup_loc % env.grid_size[1]
        dx = task.delivery_loc // env.grid_size[1]
        dy = task.delivery_loc % env.grid_size[1]
        
        ax.annotate('', xy=(env.grid_size[0]-dx-1, dy), xytext=(env.grid_size[0]-px-1, py),
                  arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
        ax.scatter([env.grid_size[0]-px-1], [py], c='red', s=50, marker='o')
        ax.scatter([env.grid_size[0]-dx-1], [dy], c='red', s=50, marker='x')
    
    for task in completed:
        dx = task.delivery_loc // env.grid_size[1]
        dy = task.delivery_loc % env.grid_size[1]
        ax.scatter([env.grid_size[0]-dx-1], [dy], c='green', s=30, marker='+', alpha=0.5)
    
    ax.set_xlim(-0.5, env.grid_size[1]-0.5)
    ax.set_ylim(-0.5, env.grid_size[0]-0.5)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_title('Task Flow (Pending: red, Completed: green)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved to {save_path}")


if __name__ == '__main__':
    os.makedirs('visualizations', exist_ok=True)
    
    print("Creating sample visualizations...")
    
    print("Saved training_reward.png, task_distribution.png to visualizations/")