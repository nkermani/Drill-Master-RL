# visualization/create_plots/methods/plot_warehouse_layout.py

"""Plot warehouse layout"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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
