# visualization/visualize/methods/plot_task_distribution.py

"""Plot task distribution"""

import matplotlib.pyplot as plt


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
