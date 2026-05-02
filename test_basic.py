# test_basic.py - Basic test to verify Drill-Master-RL works

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("Testing imports...")
try:
    from warehouse_env.warehouse import WarehouseEnv
    from warehouse_env.robot import Robot
    from warehouse_env.task import Task, TaskStatus
    print("✓ Environment imports successful")
except Exception as e:
    print(f"✗ Environment import failed: {e}")
    sys.exit(1)

try:
    from model.attention_policy import GNNEncoder, AttentionPolicy
    print("✓ Model imports successful")
except Exception as e:
    print(f"⚠ Model import failed (may need torch_geometric): {e}")
    print("  Continuing with environment tests only...")

print("\nTesting environment creation...")
try:
    env = WarehouseEnv(num_robots=5, grid_size=(10, 10), seed=42)
    print("✓ Environment created successfully")
except Exception as e:
    print(f"✗ Environment creation failed: {e}")
    sys.exit(1)

print("\nTesting environment reset...")
try:
    obs, info = env.reset()
    print(f"✓ Reset successful, obs shape: {len(obs)} robots")
    print(f"  Info: {info}")
except Exception as e:
    print(f"✗ Reset failed: {e}")
    sys.exit(1)

print("\nTesting environment step...")
try:
    actions = [0] * 5  # All robots stay
    obs, rewards, terminated, truncated, info = env.step(actions)
    print(f"✓ Step successful")
    print(f"  Rewards: {rewards}")
    print(f"  Info: {info}")
except Exception as e:
    print(f"✗ Step failed: {e}")
    sys.exit(1)

print("\n✓ Basic tests passed!")
