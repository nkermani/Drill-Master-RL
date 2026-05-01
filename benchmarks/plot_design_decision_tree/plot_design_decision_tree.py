"""Design decision flow chart"""

import matplotlib.pyplot as plt


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
