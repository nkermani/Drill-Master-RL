# N-Drill-Master-RL Technical Stack

## Overview

This project showcases proficiency with deep reinforcement learning, multi-agent systems, and graph neural networks.

---

## Technology Categories

### Deep Learning

| Technology        | Usage                          | Level     |
|------------------|--------------------------------|-----------|
| PyTorch          | Neural network framework        | Advanced  |
| PyTorch Geometric| Graph neural networks          | Advanced  |
| torch.optim      | Adam optimizer, LR scheduling   | Intermediate |
| torch.nn         | Custom layers, attention      | Advanced  |
| autograd         | Automatic differentiation     | Advanced  |

### Multi-Agent RL

| Technology      | Usage                              | Level     |
|----------------|------------------------------------|-----------|
| PPO            | Policy gradient with clipping       | Advanced  |
| QMIX           | Value function decomposition       | Advanced  |
| CTDE           | Centralized training + execution   | Advanced  |
| Replay Buffer  | Experience replay                | Intermediate |

### Graph Processing

| Technology          | Usage                       | Level     |
|--------------------|-----------------------------|-----------|
| NetworkX            | Graph utilities             | Intermediate |
| PyG Data           | Graph data structures    | Advanced    |
| GATConv            | Graph attention layers | Advanced    |
| global_mean_pool   | Graph-level aggregation | Intermediate |

### Scientific Computing

| Technology | Usage                         | Level     |
|------------|-------------------------------|-----------|
| NumPy      | Array operations, features     | Advanced  |
| Pandas     | Data analysis                | Intermediate |
| Matplotlib | Visualization              | Advanced  |
| Seaborn    | Statistical plots            | Intermediate |

### Environment

| Technology | Usage                     | Level     |
|------------|---------------------------|----------|
| Gymnasium  | RL environment interface  | Advanced |
| tqdm      | Progress bars            | Intermediate |

---

## Architecture Diagram

```
INPUT LAYER
  Robot Features  ->  (N, 6)    Node embeddings
  Edge Index     ->  (2, E)    via GNN Encoder
  Task Features ->  (T, 7)

  [Encoder] Linear(6,64) -> LayerNorm -> ReLU -> Dropout
  [GAT x3]  GATConv(64,16,heads=4) -> LayerNorm -> ReLU

POLICY HEAD
  Input:  Node embeddings (N, 64)
  State Pairs: Concat [emb_i, emb_j] -> (N, 128)
  MLP:    Linear(128, 64) -> ReLU -> Linear(64, 5)
  Output: Action probabilities (N, 5)

VALUE HEAD
  Input:  Node embeddings (N, 64)
  Global Embedding: Mean pooling
  MLP:    Linear(64, 64) -> ReLU -> Linear(64, 1)
  Output: State value (N, 1)
```

---

## Data Flow Example

```python
import torch
from src.env import WarehouseEnv
from src.model import AttentionPolicy

# 1. Create environment
env = WarehouseEnv(num_robots=10, grid_size=(10, 10), seed=42)
obs, _ = env.reset()

# 2. Extract features
robot_features = torch.tensor(obs['robot_features'], dtype=torch.float32).unsqueeze(0)
edge_index = torch.tensor([[0, 0]], dtype=torch.long)

# 3. Create policy
policy = AttentionPolicy(state_dim=6, action_dim=5, hidden_dim=64, num_agents=10)

# 4. Forward pass
action_probs, state_value = policy(robot_features, edge_index)
# Output: (1, 10, 5) action probabilities

# 5. Select actions
actions = action_probs[0].argmax(dim=-1)
# Output: tensor([0, 2, 1, 4, 3, 0, 2, 1, 4, 3])

# 6. Execute in environment
next_obs, rewards, terms, truncs, info = env.step(actions.numpy())
```

---

## Key Implementation Details

### GPU Utilization

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
policy = policy.to(device)

robot_features = robot_features.to(device)
action_probs, value = policy(robot_features, edge_index)
```

### Graph Attention Mechanism

```python
from torch_geometric.nn import GATConv

class GNNEncoder(nn.Module):
    def __init__(self):
        self.gat_convs = nn.ModuleList([
            GATConv(64, 16, heads=4, dropout=0.1)
            for _ in range(3)
        ])

    def forward(self, x, edge_index):
        for conv in self.gat_convs:
            x = conv(x, edge_index)
            x = F.elu(x)
        return x
```

### PPO Loss Computation

```python
def compute_ppo_loss(action_probs, values, returns, advantages, actions):
    dist = torch.distributions.Categorical(action_probs)
    log_probs = dist.log_prob(actions)

    # Clipped surrogate objective
    ratio = torch.exp(log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)

    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

    value_loss = F.mse_loss(values, returns)

    entropy_loss = -dist.entropy().mean()

    return policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
```

---

## Testing Coverage

| Component | Tests | Status |
|-----------|-------|--------|
| WarehouseEnv | 15 tests | ✅ |
| Robot | 5 tests | ✅ |
| Task | 4 tests | ✅ |
| AttentionPolicy | 6 tests | ✅ |
| GNNEncoder | 5 tests | ✅ |

Run with: `pytest tests/ -v`

---

## Reproducibility

```python
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.cuda.manual_seed_all(42)
```

---

*Technologies demonstrate readiness for ML engineering and research positions in deep RL*
