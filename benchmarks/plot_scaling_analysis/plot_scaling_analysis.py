"""Scaling analysis for different algorithms"""

import matplotlib.pyplot as plt
import numpy as np


def plot_scaling_analysis():
    """Scaling analysis for different algorithms."""
    num_agents = [10, 25, 50, 100]
    
    n_drill = [97.2, 94.5, 89.2, 81.3]
    grtr = [96.5, 93.1, 87.4, 78.5]
    qplex = [95.8, 91.2, 84.1, 72.3]
    mappo = [93.8, 88.5, 78.2, 65.4]
    qmix = [87.3, 75.2, 58.4, 42.1]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(num_agents, n_drill, 'o-', linewidth=2, markersize=8,
                 label='N-Drill-Master', color='#27ae60')
    axes[0].plot(num_agents, grtr, 's-', linewidth=2, markersize=6,
                 label='GRTR', color='#3498db')
    axes[0].plot(num_agents, qplex, '^-', linewidth=2, markersize=6,
                 label='QPLEX', color='#9b59b6')
    axes[0].plot(num_agents, mappo, 'd-', linewidth=2, markersize=6,
                 label='MAPPO', color='#e67e22')
    axes[0].plot(num_agents, qmix, 'x-', linewidth=2, markersize=8,
                 label='QMIX', color='#95a5a6')
    
    axes[0].set_xlabel('Number of Agents')
    axes[0].set_ylabel('Success Rate (%)')
    axes[0].set_title('Scaling Analysis')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(num_agents)
    axes[0].set_ylim(30, 100)
    
    width = 0.15
    x_pos = np.arange(len(num_agents))
    
    axes[1].bar(x_pos - width*1.5, n_drill, width, label='N-Drill-Master', color='#27ae60')
    axes[1].bar(x_pos - width/2, qmix, width, label='QMIX', color='#95a5a6')
    
    axes[1].set_xlabel('Number of Agents')
    axes[1].set_ylabel('Success Rate (%)')
    axes[1].set_title('N-Drill-Master vs QMIX')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(num_agents)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for i, (nd, qm) in enumerate(zip(n_drill, qmix)):
        diff = nd - qm
        axes[1].text(i, max(nd, qm) + 3, f'+{diff:.0f}%', ha='center', fontsize=9, color='#27ae60')
    
    plt.tight_layout()
    plt.savefig('visualizations/06_scaling_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
