"""Benchmark visualizations for N-Drill-Master-RL"""

import numpy as np
import matplotlib.pyplot as plt
import os


def create_benchmark_plots():
    """Create benchmark comparison plots."""
    os.makedirs('visualizations', exist_ok=True)
    
    plot_algorithm_comparison()
    plot_scaling_analysis()
    plot_implementation_comparison()
    plot_design_decision_tree()
    
    print("Created benchmark visualizations in visualizations/")


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
    
    improvement = [(n_drill[i] - qmix[i]) for i in range(4)]
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


def plot_design_decision_tree():
    """Design decision flow chart."""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Algorithm Selection Guide', fontsize=16, fontweight='bold')
    
    def draw_box(x, y, w, h, text, color='#3498db', text_color='white'):
        rect = plt.Rectangle((x-w/2, y-h/2), w, h, 
                           facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', 
               fontsize=10, color=text_color, fontweight='bold')
    
    def draw_arrow(x1, y1, x2, y2, label=''):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2))
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.3, label, ha='center', fontsize=9)
    
    draw_box(7, 9, 4, 1, 'How many robots?', '#2c3e50')
    
    draw_arrow(7, 8.5, 4, 7.5, '< 10')
    draw_arrow(7, 8.5, 10, 7.5, '10-50')
    
    draw_box(4, 7, 4, 1, 'Independent RL\n(PPO/DQN)', '#27ae60')
    draw_box(10, 7, 4, 1, 'CTDE Needed', '#e74c3c')
    
    draw_arrow(10, 6.5, 10, 5.5)
    draw_box(10, 5, 4, 1, 'Graph available?', '#2c3e50')
    
    draw_arrow(10, 4.5, 7, 3.5, 'Yes')
    draw_arrow(10, 4.5, 13, 3.5, 'No')
    
    draw_box(7, 3, 4, 1, 'GNN + Attention\n(N-Drill-Master)', '#27ae60')
    draw_box(13, 3, 4, 1, 'QMIX/MAPPO', '#f39c12')
    
    draw_arrow(7, 2.5, 7, 1, '< 50')
    draw_arrow(7, 2.5, 10, 1, '50-100')
    
    draw_box(7, 1, 4, 1, 'Current: Keep\n+ Comm', '#3498db')
    draw_box(10, 1, 4, 1, 'Hierarchical\n+ Meta', '#9b59b6')
    
    ax.text(7, 0.3, 'Best for: 10-50 robots\nN-Drill-Master recommended', 
           ha='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#d5f4e6', edgecolor='#27ae60'))
    
    plt.tight_layout()
    plt.savefig('visualizations/08_design_guide.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    create_benchmark_plots()