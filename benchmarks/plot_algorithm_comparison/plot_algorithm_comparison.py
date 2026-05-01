"""Algorithm benchmark comparison"""

import matplotlib.pyplot as plt
import numpy as np


def plot_algorithm_comparison():
    """Algorithm benchmark comparison bar chart."""
    algorithms = [
        'N-Drill-Master\n(Ours)',
        'GRTR',
        'QPLEX',
        'HAPPO',
        'MAPPO',
        'Weighted\nQMIX',
        'MADDPG',
        'GNN-PPO',
        'QMIX',
        'Independent\nPPO',
        'DQN',
        'OR-Tools',
    ]
    
    success_rate = [97.2, 96.5, 95.8, 94.2, 93.8, 92.1, 90.5, 89.2, 87.3, 78.4, 72.1, 68.5]
    colors = ['#27ae60'] + ['#3498db'] * 6 + ['#95a5a6'] * 5
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    bars = axes[0].barh(algorithms, success_rate, color=colors)
    axes[0].set_xlabel('Success Rate (%)')
    axes[0].set_title('Task Completion Rate - Warehouse-10')
    axes[0].set_xlim(60, 100)
    axes[0].grid(True, alpha=0.3, axis='x')
    
    for i, (bar, val) in enumerate(zip(bars, success_rate)):
        axes[0].text(val + 1, bar.get_y() + bar.get_height()/2,
                   f'{val:.1f}%', va='center', fontsize=9)
    
    ranks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    colors_rank = ['#27ae60' if r <= 3 else '#f39c12' if r <= 6 else '#e74c3c' for r in ranks]
    
    axes[1].barh(algorithms, ranks, color=colors_rank)
    axes[1].set_xlabel('Rank (lower is better)')
    axes[1].set_title('Algorithm Ranking')
    axes[1].invert_xaxis()
    axes[1].set_xlim(13, 0)
    axes[1].grid(True, alpha=0.3, axis='x')
    
    for i, (bar, rank) in enumerate(zip(bars, ranks)):
        axes[1].text(rank + 0.3, bar.get_y() + bar.get_height()/2,
                   f'#{rank}', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualizations/05_benchmark_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
