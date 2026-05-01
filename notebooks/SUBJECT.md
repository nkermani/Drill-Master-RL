# N-Drill-Master-RL

*Reinforcement Learning for Multi-Robot Fleet Navigation under Uncertainty*

## Overview

N-Drill-Master-RL is a research-oriented project exploring the intersection of Deep Reinforcement Learning (Deep RL) and Multi-Agent Systems for robot fleet management. Inspired by the challenges at NAVER LABS Europe on Neural Combinatorial Optimization for Robot Fleet Management, this project implements RL-based solutions for pick-up and delivery services involving robot fleets moving in dynamic, uncertain environments.

The project simulates a fleet of warehouse robots navigating a grid-based facility with:
- **Dynamic task arrivals**: Pick-up and delivery requests appear over time
- **Uncertainty**: Task durations and robot speeds vary
- **Multi-agent coordination**: Robots must avoid collisions and optimize global throughput

## The Challenge: RL meets Combinatorial Optimization

Traditional approaches to multi-robot routing:
| Approach | Pros | Cons |
|---------|-----|-----|
| **Exact Optimization (MILP)** | Optimal solutions | Doesn't scale, no real-time adaptation |
| **Heuristics (OR-Tools)** | Fast, adjustable | Requires domain expertise, brittle |
| **Standard RL (PPO, DQN)** | Learns from experience | Doesn't scale to many agents |
| **Multi-Agent RL (QMIX, MAPPO)** | Cooperative agents | Credit assignment challenge |

N-Drill-Master-RL addresses these by implementing:
- **Centralized Training with Decentralized Execution (CTDE)**
- **Graph-based state representations** for scalable coordination
- **Attention mechanisms** for dynamic agent interactions

## Technical Stack

| Component | Technology |
|-----------|------------|
| Deep Learning Framework | PyTorch |
| Multi-Agent RL | PyTorch (custom implementations) |
| Graph Neural Networks | PyTorch Geometric |
| Environment | Custom Gym-compatible environment |
| Visualization | Matplotlib, NumPy |

### Architecture

- **Encoder**: Embeds robot states, task features, and spatial relationships
- **Processor**: GNN layers for inter-agent message passing
- **Decoder**: Policy heads for action selection (movement, pickup, delivery)
- **Value Head**: Centralized value function for CTDE

## How it Works

1. **State Representation**: The environment is modeled as a graph where:
   - Nodes = locations (stations, robot positions)
   - Edges = navigable paths between locations
   - Features = task queue, robot load, time urgency

2. **Neural Policy**: A graph attention network processes the state:
   - Robot embeddings capture individual states
   - Attention aggregates information from neighbors
   - Actor outputs action logits (stay, move, pickup, deliver)

3. **Training**: PPO-style policy gradient with:
   -熵 regularization for exploration
   - Value function for advantage estimation
   -_clipping for training stability

4. **Multi-Agent Coordination**: QMIX-style mixing network combines per-agent values into a global Qtot:

$$Q_{tot} = \phi^{-1}\left(\sum_i \phi(Q_i, \tau_i)\right)$$

## Research Sell

> "This project demonstrates the ability to bridge reinforcement learning and combinatorial optimization. By leveraging Graph Neural Networks for inter-agent communication and centralized training for decentralized execution, N-Drill-Master-RL provides a scalable solution for multi-agent coordination in dynamic, uncertain environments—directly addressing the design bottlenecks of robot fleet management."

## Project Structure

```
N-Drill-Master-RL/
├── data/                   # Generated task datasets
├── notebooks/              # Exploratory analysis
├── src/
│   ├── env/               # Multi-agent environment
│   │   ├── warehouse.py   # Warehouse layout generator
│   │   ├── robot.py       # Robot agent class
│   │   └── task.py        # Task generation logic
│   ├── model/             # RL and GNN models
│   │   ├── attention_policy.py
│   │   ├── qmix.py
│   │   └── gnn_encoder.py
│   └── train.py           # Training loops
├── tests/                 # Unit tests
└── README.md
```

## Connection to NAVER LABS Europe

This project directly implements the research track described in the NAVER LABS Europe internship:
- Reinforcement Learning and planning
- Multi-agent systems under uncertainty
- Learning-augmented optimization
- Graph neural networks for robot coordination

---

*Last Updated: April 2026*
*Author: Nathan Kermani*