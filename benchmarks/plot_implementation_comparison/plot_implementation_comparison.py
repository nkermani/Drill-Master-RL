"""Implementation complexity comparison"""

import matplotlib.pyplot as plt
import numpy as np


def plot_implementation_comparison():
    """Implementation complexity comparison."""
    algorithms = ['QMIX', 'MADDPG', 'MAPPO', 'N-Drill-Master', 'QPLEX', 'GRTR']
    
    params = [1.5, 2.8, 2.8, 2.1, 4.8, 3.2]
    gpu_mem = [3.2, 5.5, 5.5, 4.2, 8.5, 6.1]
    train_hours = [3, 5, 5, 4, 8, 6]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    x = np.arange(len(algorithms))
    width = 0.25
    
    colors = ['#95a5a6', '#e67e22', '#9b59b6', '#27ae60', '#3498db', '#e74c3c']
    
    axes[0, 0].bar(x, params, color=colors)
    axes[0, 0].set_ylabel('Parameters (M)')
    axes[0, 0].set_title('Model Size')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(algorithms, rotation=45, ha='right')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    for i, p in enumerate(params):
        axes[0, 0].text(i, p + 0.1, f'{p:.1f}M', ha='center', fontsize=8)
    
    axes[0, 1].bar(x, gpu_mem, color=colors)
    axes[0, 1].set_ylabel('GPU Memory (GB)')
    axes[0, 1].set_title('GPU Memory')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(algorithms, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    for i, m in enumerate(gpu_mem):
        axes[0, 1].text(i, m + 0.2, f'{m:.1f}GB', ha='center', fontsize=8)
    
    axes[1, 0].bar(x, train_hours, color=colors)
    axes[1, 0].set_ylabel('Training Time (hours)')
    axes[1, 0].set_title('Training Time')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(algorithms, rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    for i, t in enumerate(train_hours):
        axes[1, 0].text(i, t + 0.2, f'{t}h', ha='center', fontsize=8)
    
    speedup = [train_hours[0] / t for t in train_hours]
    axes[1, 1].bar(x, speedup, color=colors)
    axes[1, 1].set_ylabel('Speedup vs QMIX')
    axes[1, 1].set_title('Relative Training Speed')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(algorithms, rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    
    for i, s in enumerate(speedup):
        axes[1, 1].text(i, s + 0.1, f'{s:.1f}x', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('visualizations/07_implementation_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
