# visualization_scripts/create_plots/methods/plot_gnn_attention.py

"""Plot GNN attention"""

import numpy as np
import matplotlib.pyplot as plt


def plot_gnn_attention():
    """Sample GNN attention heatmap."""
    np.random.seed(42)
    num_agents = 10

    attention = np.random.rand(num_agents, num_agents)
    attention = (attention + attention.T) / 2
    np.fill_diagonal(attention, 0)
    attention = attention / attention.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    im = axes[0].imshow(attention, cmap='YlOrRd', aspect='auto')
    axes[0].set_xlabel('To Robot')
    axes[0].set_ylabel('From Robot')
    axes[0].set_title('GNN Attention Weights')
    axes[0].set_xticks(range(num_agents))
    axes[0].set_yticks(range(num_agents))
    plt.colorbar(im, ax=axes[0])

    layers = ['Input', 'GAT-1', 'GAT-2', 'GAT-3', 'Policy']
    values = [6, 64, 64, 64, 5]
    axes[1].bar(layers, values, color=['#3498db', '#2ecc71', '#2ecc71', '#2ecc71', '#e74c3c'])
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Hidden Dim')
    axes[1].set_title('Network Architecture')
    axes[1].grid(True, alpha=0.3, axis='y')

    for i, v in enumerate(values):
        axes[1].text(i, v + 2, str(v), ha='center')

    plt.tight_layout()
    plt.savefig('visualizations/04_gnn_attention.png', dpi=150)
    plt.close()
