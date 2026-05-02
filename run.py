# run.py

#!/usr/bin/env python3
"""
Run script for N-Drill-Master-RL
Tests environment, generates visuals, and trains model
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported"""
    print("=" * 50)
    print("Testing imports...")
    print("=" * 50)

    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False

    try:
        import gymnasium
        print(f"✓ Gymnasium {gymnasium.__version__}")
    except ImportError as e:
        print(f"✗ Gymnasium import failed: {e}")
        return False

    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False

    return True


def test_environment():
    """Test the warehouse environment"""
    print("\n" + "=" * 50)
    print("Testing Warehouse Environment...")
    print("=" * 50)

    try:
        from src import WarehouseEnv
        env = WarehouseEnv(
            num_robots=5,
            grid_size=(8, 8),
            num_stations=4,
            task_arrival_rate=0.3,
            max_tasks=20,
            seed=42
        )
        obs, info = env.reset()
        print(f"✓ Environment created successfully")
        print(f"  - Robots: {len(env.robots)}")
        print(f"  - Grid size: {env.grid_size}")
        print(f"  - Stations: {len(env.stations)}")
        print(f"  - Active tasks: {info['active_tasks']}")
        return env
    except ImportError as e:
        print(f"✗ Environment import failed: {e}")
        print("  Note: WarehouseEnv may not be implemented yet")
        return None
    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        return None


def generate_visuals(env=None):
    """Generate visualization plots"""
    print("\n" + "=" * 50)
    print("Generating Visuals...")
    print("=" * 50)

    os.makedirs('visualizations', exist_ok=True)

    try:
        from visualization_scripts.create_plots.methods.create_sample_plots import create_sample_plots
        create_sample_plots()
        print("✓ Sample plots created in visualizations/")
    except Exception as e:
        print(f"✗ Failed to create sample plots: {e}")

    if env:
        try:
            from visualization_scripts.visualize.methods.visualize_warehouse import visualize_warehouse
            visualize_warehouse(env, save_path='visualizations/warehouse_env.png')
            print("✓ Warehouse visualization saved")
        except Exception as e:
            print(f"✗ Failed to visualize warehouse: {e}")


def train_model(env=None):
    """Train the RL model"""
    print("\n" + "=" * 50)
    print("Training Model...")
    print("=" * 50)

    try:
        from src import AttentionPolicy, Trainer

        if env is None:
            from src import WarehouseEnv
            env = WarehouseEnv(num_robots=5, grid_size=(8, 8), seed=42)

        print("Creating policy network...")
        policy = AttentionPolicy(
            state_dim=6,
            action_dim=5,
            hidden_dim=64,
            num_agents=5
        )

        print("Setting up trainer...")
        trainer = Trainer(
            env=env,
            policy=policy,
            value_network=policy,
            num_agents=5,
            learning_rate=3e-4,
            batch_size=32,
            max_steps=100
        )

        print("Starting training (100 updates)...")
        history = trainer.train(num_updates=100, log_interval=20)
        print(f"✓ Training complete!")
        print(f"  - Final reward: {history['total_reward'][-1]:.2f}")

        from src.train.visualize_training import visualize_training
        visualize_training(history, save_path='visualizations/training_curves.png')
        print("✓ Training curves saved")

        return history
    except ImportError as e:
        print(f"✗ Training import failed: {e}")
        print("  Note: Required modules may not be implemented yet")
        return None
    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("\nN-Drill-Master-RL: Running Tests, Visuals, and Training\n")

    if not test_imports():
        print("\n✗ Import test failed. Please check dependencies.")
        sys.exit(1)

    env = test_environment()
    generate_visuals(env)

    if env:
        train_model(env)
    else:
        print("\n⚠ Skipping training - environment not available")

    print("\n" + "=" * 50)
    print("Run complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
