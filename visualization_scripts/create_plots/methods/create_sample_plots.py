# visualization_scripts/create_plots/methods/create_sample_plots.py

"""Create sample plots entry point"""

import os
from .plot_training_progress import plot_training_progress
from .plot_robot_performance import plot_robot_performance
from .plot_warehouse_layout import plot_warehouse_layout
from .plot_gnn_attention import plot_gnn_attention


def create_sample_plots():
    """Create sample visualization plots."""
    os.makedirs('visualizations', exist_ok=True)

    plot_training_progress()
    plot_robot_performance()
    plot_warehouse_layout()
    plot_gnn_attention()

    print("Created visualization files in visualizations/")
