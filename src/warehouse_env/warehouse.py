# src/warehouse_env/warehouse.py

"""Gymnasium-compatible multi-agent warehouse environment"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Any, Tuple
from .robot import Robot
from .task import Task, TaskStatus, generate_task


class WarehouseEnv(gym.Env):
    """Multi-agent warehouse environment for robot fleet management"""

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}

    def __init__(
        self,
        num_robots: int = 10,
        grid_size: Tuple[int, int] = (10, 10),
        num_stations: int = 8,
        task_arrival_rate: float = 0.3,
        max_tasks: int = 50,
        max_steps: int = 1000,
        seed: Optional[int] = None
    ):
        super().__init__()

        self.num_robots = num_robots
        self.grid_size = grid_size
        self.num_stations = num_stations
        self.task_arrival_rate = task_arrival_rate
        self.max_tasks = max_tasks
        self.max_steps = max_steps

        self.np_random = None
        self.seed(seed)

        self.stations = []
        self.robots = []
        self.tasks = {}
        self.task_counter = 0
        self.current_step = 0

        self.action_space = spaces.Tuple([
            spaces.Discrete(5) for _ in range(num_robots)
        ])

        self.observation_space = spaces.Tuple([
            spaces.Box(low=0, high=max(grid_size), shape=(6,), dtype=np.float32)
            for _ in range(num_robots)
        ])

        self._initialize_stations()
        self.reset()

    def seed(self, seed: Optional[int] = None):
        self.np_random = np.random.RandomState(seed)

    def _initialize_stations(self):
        """Initialize station locations on the grid"""
        self.stations = []
        margin = 1
        for i in range(self.num_stations):
            x = self.np_random.randint(margin, self.grid_size[0] - margin)
            y = self.np_random.randint(margin, self.grid_size[1] - margin)
            self.stations.append([x, y])

    def _initialize_robots(self):
        """Initialize robots at random positions"""
        self.robots = []
        for i in range(self.num_robots):
            pos = np.array([
                self.np_random.randint(0, self.grid_size[0]),
                self.np_random.randint(0, self.grid_size[1])
            ], dtype=np.float32)
            robot = Robot(id=i, position=pos, state='idle')
            self.robots.append(robot)

    def reset(self, seed: Optional[int] = None) -> Tuple[tuple, Dict]:
        if seed is not None:
            self.seed(seed)

        self.tasks = {}
        self.task_counter = 0
        self.current_step = 0

        self._initialize_robots()
        self._initialize_stations()

        obs = tuple(robot.get_state_vector() for robot in self.robots)
        info = {
            'active_tasks': 0,
            'completed_tasks': 0,
            'total_reward': 0.0
        }

        return obs, info

    def step(self, actions: List[int]) -> Tuple[tuple, List[float], bool, bool, Dict]:
        self.current_step += 1
        rewards = []

        for i, robot in enumerate(self.robots):
            action = actions[i]
            reward = self._execute_action(robot, action)
            rewards.append(reward)

        self._generate_tasks()
        self._update_robots()

        obs = tuple(robot.get_state_vector() for robot in self.robots)

        # Don't terminate in first few steps to allow tasks to be generated
        if self.current_step < 10:
            terminated = False
        else:
            terminated = all(robot.state == 'idle' for robot in self.robots) and len(self._get_active_tasks()) == 0
        truncated = self.current_step >= self.max_steps

        info = {
            'active_tasks': len(self._get_active_tasks()),
            'completed_tasks': len(self._get_completed_tasks()),
            'total_reward': sum(rewards)
        }

        return obs, rewards, terminated, truncated, info

    def _execute_action(self, robot: Robot, action: int) -> float:
        reward = -0.01

        if action == 0:  # Stay
            pass
        elif action == 1:  # Move to pickup or deliver
            if robot.current_task is not None:
                task = self.tasks.get(robot.current_task)
                if task:
                    if task.status == TaskStatus.ASSIGNED:
                        # Moving to pickup
                        reached = robot.move_towards(task.pickup_location)
                        if reached:
                            task.start()
                            reward += 1.0
                    elif task.status == TaskStatus.IN_PROGRESS:
                        # Moving to delivery
                        reached = robot.move_towards(task.delivery_location)
                        if reached:
                            robot.deliver()
                            task.complete(self.current_step)
                            reward += 10.0
        elif action == 2:  # Pickup task
            if robot.can_pickup():
                active_tasks = self._get_active_tasks()
                if active_tasks:
                    task = min(active_tasks, key=lambda t: np.linalg.norm(robot.position - t.pickup_location))
                    task.assign(robot.id, self.current_step)
                    robot.pickup(task.id)
                    reward += 0.5
        elif action == 3:  # Deliver task
            if robot.current_task is not None:
                task = self.tasks.get(robot.current_task)
                if task and task.status == TaskStatus.IN_PROGRESS:
                    reached = robot.move_towards(task.delivery_location)
                    if reached:
                        robot.deliver()
                        task.complete(self.current_step)
                        reward += 10.0
        elif action == 4:  # Charge
            if robot.battery < 50:
                robot.battery = min(robot.max_battery, robot.battery + 10)
                reward -= 0.5

        robot.update_battery()
        return reward

    def _generate_tasks(self):
        if len(self._get_active_tasks()) < self.max_tasks:
            if self.np_random.random() < self.task_arrival_rate:
                task = generate_task(
                    self.task_counter,
                    self.current_step,
                    self.grid_size,
                    self.num_stations,
                    self.stations
                )
                self.tasks[self.task_counter] = task
                self.task_counter += 1

    def _update_robots(self):
        for robot in self.robots:
            if robot.current_task:
                task = self.tasks.get(robot.current_task)
                if task and task.status == TaskStatus.COMPLETED:
                    robot.deliver()

    def _get_active_tasks(self) -> List[Task]:
        return [t for t in self.tasks.values() if t.is_active]

    def _get_completed_tasks(self) -> List[Task]:
        return [t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED]

    def render(self, mode: str = 'human'):
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Active tasks: {len(self._get_active_tasks())}")
            print(f"Completed tasks: {len(self._get_completed_tasks())}")
            for robot in self.robots:
                print(f"Robot {robot.id}: pos={robot.position}, state={robot.state}, load={robot.load}")

    def close(self):
        pass
