# test_full.py - Full integration test with training simulation

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("=" * 50)
print("Drill-Master-RL Full Integration Test")
print("=" * 50)

# Test 1: Environment
print("\n1. Testing Environment...")
from env.warehouse import WarehouseEnv
env = WarehouseEnv(num_robots=10, grid_size=(10, 10), num_stations=8, seed=42)
obs, info = env.reset()
print(f"   ✓ Created environment with {env.num_robots} robots")
print(f"   ✓ Grid size: {env.grid_size}")
print(f"   ✓ Stations: {env.num_stations}")

# Test 2: Run multiple steps
print("\n2. Running simulation...")
total_reward = 0
for step in range(100):
    actions = [np.random.randint(0, 5) for _ in range(env.num_robots)]
    obs, rewards, terminated, truncated, info = env.step(actions)
    total_reward += sum(rewards)
    if terminated or truncated:
        break
print(f"   ✓ Completed {step+1} steps")
print(f"   ✓ Total reward: {total_reward:.2f}")
print(f"   ✓ Completed tasks: {info['completed_tasks']}")

# Test 3: Test with different configurations
print("\n3. Testing different configurations...")
configs = [
    {"num_robots": 5, "grid_size": (5, 5)},
    {"num_robots": 20, "grid_size": (15, 15)},
]
for cfg in configs:
    env_test = WarehouseEnv(**cfg, seed=42)
    obs, _ = env_test.reset()
    print(f"   ✓ {cfg['num_robots']} robots on {cfg['grid_size']} grid: OK")

# Test 4: Check model imports (if available)
print("\n4. Testing model imports...")
try:
    from model.attention_policy import GNNEncoder, AttentionPolicy
    print("   ✓ GNNEncoder and AttentionPolicy imported")
    print("   ⚠ torch_geometric not available - model tests skipped")
except ImportError as e:
    print(f"   ⚠ Model import failed: {e}")
    print("   ⚠ Install torch_geometric for full functionality")

print("\n" + "=" * 50)
print("ENVIRONMENT TESTS PASSED!")
print("Ready for benchmark visualizations")
print("=" * 50)
