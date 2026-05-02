# N-Drill-Master-RL

*Reinforcement Learning for Multi-Robot Fleet Navigation under Uncertainty*

## Overview

N-Drill-Master-RL implements deep reinforcement learning solutions for multi-robot fleet management in dynamic warehouse environments. The project explores the intersection of Deep RL and combinatorial optimization, addressing challenges in pick-up and delivery services with robot fleets operating under uncertainty.

## Features

- **Multi-Agent RL**: CTDE (Centralized Training with Decentralized Execution)
- **Graph Attention Networks**: Scalable inter-agent communication
- **PPO-style Policy Gradient**: Stable training with clipped objectives
- **QMIX-style Value Mixing**: Cooperative multi-agent learning
- **Gymnasium-Compatible Environment**: Easy integration with RL frameworks

## Installation

### Quick Setup with Nix

This project uses Nix for reproducible environment management. To get started:

```bash
git clone https://github.com/nkermani/N-Drill-Master-RL.git
cd N-Drill-Master-RL

# Enter the Nix shell (automatically creates venv and installs dependencies)
nix-shell

# The shell will:
# 1. Create a Python 3.11 virtual environment in .venv/
# 2. Install PyTorch and all requirements
# 3. Activate the environment
#
# To exit: deactivate
# To re-enter: nix-shell
```

### Manual Installation (without Nix)

```bash
git clone https://github.com/nkermani/N-Drill-Master-RL.git
cd N-Drill-Master-RL
pip install -r requirements.txt
```

**Dependencies:**
- PyTorch >= 2.0
- PyTorch Geometric >= 2.3
- Gymnasium >= 0.29
- NumPy, Pandas, Matplotlib

## Quick Start

### 1. Quick Start with Nix

After entering the nix-shell (see Installation above):

```bash
# Run the complete pipeline: environment test, visualization, and training
python run.py
```

This will:
- Test all imports (PyTorch, Gymnasium, Matplotlib)
- Create the warehouse environment
- Generate visualization plots in `visualizations/`
- Train the RL model for 100 updates

### 2. Create a Warehouse Environment

```python
from src.env import WarehouseEnv

env = WarehouseEnv(
    num_robots=10,
    grid_size=(10, 10),
    num_stations=8,
    task_arrival_rate=0.3,
    max_tasks=50,
    seed=42
)

obs, info = env.reset()
print(f"Active tasks: {info['active_tasks']}")
```

### 2. Run the Environment

```python
actions = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]  # Example actions per robot
obs, rewards, terminations, truncations, info = env.step(actions)

print(f"Rewards: {rewards}")
print(f"Info: {info}")
```

### 3. Train the RL Agent

```python
import torch
from src.model import AttentionPolicy, GNNEncoder
from src.train import Trainer

policy = AttentionPolicy(
    state_dim=6,
    action_dim=5,
    hidden_dim=64,
    num_agents=10
)

value_network = AttentionPolicy(
    state_dim=6,
    action_dim=5,
    hidden_dim=64,
    num_agents=10
)

trainer = Trainer(
    env=env,
    policy=policy,
    value_network=value_network,
    num_agents=10,
    learning_rate=3e-4,
    batch_size=64,
    max_steps=1000
)

history = trainer.train(num_updates=1000, log_interval=10)

print(f"Final reward: {history['total_reward'][-1]:.2f}")
```

### 4. Save/Load Checkpoints

```python
trainer.save_checkpoint('checkpoints/model.pt')
trainer.load_checkpoint('checkpoints/model.pt')
```

## Project Structure

```
N-Drill-Master-RL/
├── data/                   # Generated datasets
├── notebooks/              # Exploratory analysis
├── src/
│   ├── env/               # Multi-agent environment
│   │   ├── warehouse.py   # Warehouse environment
│   │   └── __init__.py
│   ├── model/             # RL and GNN models
│   │   ├── attention_policy.py
│   │   ├── qmix.py
│   │   └── __init__.py
│   ├── train.py           # Training loops
│   └── __init__.py
├── tests/                 # Unit tests
├── requirements.txt
├── README.md
├── SUBJECT.md
├── TECHNICAL_STACK.md
└── EXPLANATIONS.md
```

## API Reference

### Environment

| Class | Description |
|-------|-------------|
| `WarehouseEnv` | Gymnasium-compatible multi-agent environment |
| `Robot` | Individual robot agent |
| `Task` | Pick-up/delivery task |

### Model

| Class | Description |
|-------|-------------|
| `GNNEncoder` | Graph Attention Network encoder |
| `AttentionPolicy` | Policy network with GAT backbone |
| `CentralizedCritic` | Centralized value function |
| `MixingNetwork` | QMIX-style mixing network |

### Training

| Class | Description |
|-------|-------------|
| `Trainer` | PPO-style training loop |
| `PPOAgent` | PPO agent with advantage estimation |
| `ReplayBuffer` | Experience replay buffer |

## Running Tests

```bash
# Inside nix-shell
pytest tests/ -v
```

## Visualization

```bash
# Inside nix-shell
python -c "
from src.train import visualize_training
history = {'policy_loss': [0.5], 'value_loss': [0.3], 'entropy': [1.2], 'total_reward': [100]}
visualize_training(history, save_path='training_curves.png')
"
```

## Complete Pipeline

```bash
# Enter nix-shell (first time: creates venv and installs everything)
nix-shell

# Run everything: test environment, create visuals, train model
python run.py

# Exit when done
deactivate
# Re-enter anytime with: nix-shell
```

---

*Last Updated: April 2026*
*Author: Nathan Kermani*