# generate_benchmark_data.py - Generate data and visuals for benchmarks

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from env.warehouse import WarehouseEnv
from env.robot import Robot
from env.task import Task, TaskStatus


def smart_policy(robot: Robot, tasks: Dict[int, Task]) -> int:
    """Simple heuristic policy for task completion"""
    # If robot has a task in progress, move to deliver
    if robot.current_task is not None:
        task = tasks.get(robot.current_task)
        if task and task.status == TaskStatus.IN_PROGRESS:
            return 3  # Deliver
        elif task and task.status == TaskStatus.ASSIGNED:
            return 1  # Move to pickup

    # If idle and can pickup, find nearest task
    if robot.state == 'idle' and robot.load == 0:
        active_tasks = [t for t in tasks.values() if t.status == TaskStatus.PENDING]
        if active_tasks:
            return 2  # Pickup

    # Default: stay
    return 0


def run_smart_simulation(num_robots: int, grid_size: tuple, steps: int = 500, seed: int = 42) -> Dict:
    """Run a simulation with smart policy and return metrics"""
    env = WarehouseEnv(num_robots=num_robots, grid_size=grid_size, seed=seed)
    obs, info = env.reset()

    total_rewards = []
    completed_tasks = []
    active_tasks = []

    for step in range(steps):
        actions = []
        for robot in env.robots:
            action = smart_policy(robot, env.tasks)
            actions.append(action)

        obs, rewards, terminated, truncated, info = env.step(actions)
        total_rewards.append(sum(rewards))
        completed_tasks.append(info['completed_tasks'])
        active_tasks.append(info['active_tasks'])

        if terminated or truncated:
            break

    return {
        'total_rewards': total_rewards,
        'completed_tasks': completed_tasks,
        'active_tasks': active_tasks,
        'final_completed': completed_tasks[-1] if completed_tasks else 0,
        'avg_reward': np.mean(total_rewards) if total_rewards else 0,
        'steps': len(total_rewards)
    }


def plot_benchmark_results(results: List[Dict], labels: List[str], save_path: str):
    """Create benchmark comparison plot"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Task completion by configuration
    for i, (result, label) in enumerate(zip(results, labels)):
        axes[0, 0].plot(result['completed_tasks'], label=label, linewidth=2)
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Completed Tasks')
    axes[0, 0].set_title('Task Completion Over Time (Smart Policy)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Active tasks
    for i, (result, label) in enumerate(zip(results, labels)):
        axes[0, 1].plot(result['active_tasks'], label=label, linewidth=2)
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Active Tasks')
    axes[0, 1].set_title('Active Tasks Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Cumulative rewards
    for i, (result, label) in enumerate(zip(results, labels)):
        cum_rewards = np.cumsum(result['total_rewards'])
        axes[1, 0].plot(cum_rewards, label=label, linewidth=2)
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Cumulative Reward')
    axes[1, 0].set_title('Cumulative Reward Over Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Final performance comparison
    final_completed = [r['final_completed'] for r in results]
    x = np.arange(len(labels))

    bars = axes[1, 1].bar(x, final_completed, 0.5, color='steelblue', alpha=0.7)
    axes[1, 1].set_xlabel('Configuration')
    axes[1, 1].set_ylabel('Tasks Completed')
    axes[1, 1].set_title('Final Performance Comparison')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels, rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, value in zip(bars, final_completed):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(value), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved benchmark plot to {save_path}")


def main():
    print("=" * 50)
    print("Generating Benchmark Data and Visualizations")
    print("=" * 50)

    os.makedirs('visualizations', exist_ok=True)

    # Test different configurations
    configs = [
        {'num_robots': 5, 'grid_size': (10, 10), 'label': '5 Robots'},
        {'num_robots': 10, 'grid_size': (10, 10), 'label': '10 Robots'},
        {'num_robots': 20, 'grid_size': (15, 15), 'label': '20 Robots'},
    ]

    results = []
    labels = []

    for cfg in configs:
        label = cfg.pop('label')
        print(f"\nRunning simulation: {label}")
        result = run_smart_simulation(steps=1000, seed=42, **cfg)
        results.append(result)
        labels.append(label)
        print(f"  ✓ Completed {result['final_completed']} tasks in {result['steps']} steps")
        print(f"  ✓ Average reward: {result['avg_reward']:.2f}")

    # Create benchmark plot
    plot_benchmark_results(results, labels, 'visualizations/benchmark_results.png')

    print("\n" + "=" * 50)
    print("BENCHMARK VISUALS READY!")
    print("=" * 50)
    print("\nGenerated files:")
    print("  - visualizations/benchmark_results.png")
    print("  - visualizations/01_training_progress.png")
    print("  - visualizations/05_benchmark_comparison.png")


if __name__ == '__main__':
    main()
