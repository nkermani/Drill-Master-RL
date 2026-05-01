# visualization/visualize/methods/visualize_warehouse.py

"""Visualize warehouse environment"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def visualize_warehouse(env, save_path=None):
    """Visualize the warehouse grid with robots and tasks."""
    fig, ax = plt.subplots(figsize=(10, 10))

    grid = [[0 for _ in range(env.grid_size[1])] for _ in range(env.grid_size[0])]

    for x in range(env.grid_size[0]):
        for y in range(env.grid_size[1]):
            idx = x * env.grid_size[1] + y
            if env.grid[x, y] == 1:
                grid[x][y] = 0.3

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
    ax.set_yticklabels(list(range(env.grid_size[0], reversed(range(env.grid_size[0])))
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
