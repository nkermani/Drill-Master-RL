"""Create benchmark plots entry point"""

import os


def create_benchmark_plots():
    """Create benchmark comparison plots."""
    os.makedirs('visualizations', exist_ok=True)
    
    plot_algorithm_comparison()
    plot_scaling_analysis()
    plot_implementation_comparison()
    plot_design_decision_tree()
    
    print("Created benchmark visualizations in visualizations/")
