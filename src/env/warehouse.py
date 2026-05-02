# src/env/warehouse.py

"""Warehouse environment for multi-robot fleet management"""
import numpy as np
from enum import Enum
from gymnasium import Env, spaces


class RobotState(Enum):
    IDLE = 0
    MOVING = 1
    CARRYING = 2
    CHARGING = 3


class Task:
    def __init__(self, task_id, pickup_pos, delivery_pos, creation_step):
        self.task_id = task_id
        self.pickup_pos = pickup_pos
        self.delivery_pos = delivery_pos
        self.creation_step = creation_step
        self.assigned_robot = None
        self.status = 'pending'


class Robot:
    def __init__(self, robot_id, start_pos, grid_size):
        self.robot_id = robot_id
        self.pos = start_pos
        self.grid_size = grid_size
        self.state = RobotState.IDLE
        self.current_task = None
        self.battery = 100.0

    def move(self, direction):
        """Move robot in given direction (0: up, 1: right, 2: down, 3: left, 4: stay)"""
        x, y = self.pos // self.grid_size[1], self.pos % self.grid_size[1]
        if direction == 0 and x > 0:
            x -= 1
        elif direction == 1 and y < self.grid_size[1] - 1:
            y += 1
        elif direction == 2 and x < self.grid_size[0] - 1:
            x += 1
        elif direction == 3 and y > 0:
            y -= 1
        self.pos = x * self.grid_size[1] + y
        self.state = RobotState.MOVING


class WarehouseEnv(Env):
    """Multi-robot warehouse environment"""

    def __init__(self, num_robots=5, grid_size=(10, 10), num_stations=4,
                 task_arrival_rate=0.3, max_tasks=50, seed=42):
        super().__init__()
        self.num_robots = num_robots
        self.grid_size = grid_size
        self.num_stations = num_stations
        self.task_arrival_rate = task_arrival_rate
        self.max_tasks = max_tasks
        self.seed = seed
        self.np_random = np.random.RandomState(seed)

        self.grid = np.zeros(grid_size)
        self.stations = self._generate_stations()
        self.robots = [Robot(i, i * grid_size[1] // num_robots, grid_size)
                       for i in range(num_robots)]
        self.tasks = {}
        self.current_step = 0
        self.task_counter = 0

        self.action_space = spaces.MultiDiscrete([5] * num_robots)
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(num_robots, 6), dtype=np.float32)

    def _generate_stations(self):
        stations = []
        for i in range(self.num_stations):
            x = self.np_random.randint(0, self.grid_size[0])
            y = self.np_random.randint(0, self.grid_size[1])
            stations.append(x * self.grid_size[1] + y)
        return stations

    def reset(self):
        self.current_step = 0
        self.tasks = {}
        self.task_counter = 0
        for robot in self.robots:
            robot.pos = robot.robot_id * self.grid_size[1] // self.num_robots
            robot.state = RobotState.IDLE
            robot.current_task = None
        obs = {'robot_features': self._get_observations()}
        info = {'active_tasks': len(self.tasks)}
        return obs, info

    def step(self, actions):
        self.current_step += 1
        rewards = []

        for i, action in enumerate(actions):
            robot = self.robots[i]
            if action < 4:
                robot.move(action)
            reward = -0.01
            if self.np_random.random() < self.task_arrival_rate:
                self._spawn_task()
            rewards.append(reward)

        obs = {'robot_features': self._get_observations()}
        terminations = [False] * self.num_robots
        truncations = [False] * self.num_robots
        info = {'active_tasks': len(self.tasks)}

        return obs, rewards, terminations, truncations, info

    def _spawn_task(self):
        if len(self.tasks) >= self.max_tasks:
            return
        pickup = self.np_random.randint(0, self.grid_size[0] * self.grid_size[1])
        delivery = self.np_random.choice(self.stations)
        task = Task(self.task_counter, pickup, delivery, self.current_step)
        self.tasks[self.task_counter] = task
        self.task_counter += 1

    def _get_observations(self):
        obs = []
        for robot in self.robots:
            state = [
                robot.pos / (self.grid_size[0] * self.grid_size[1]),
                robot.state.value / 3.0,
                self.tasks.get(0, None) is not None,
                0.5,
                0.5,
                robot.battery / 100.0
            ]
            obs.append(state)
        return np.array(obs, dtype=np.float32)
