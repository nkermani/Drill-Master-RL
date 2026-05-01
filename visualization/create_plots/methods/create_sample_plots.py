# visualization/create_plots/methods/create_sample_plots.py

"""Create sample plots entry point"""

import os


def create_sample_plots():
    """Create sample visualization plots."""
    os.makedirs('visualizations', exist_ok=True)

    plot_training_progress()
    plot_robot_performance()
    plot_warehouse_layout()
    plot_gnn_attention()

    print("Created visualization files in visualizations/")
