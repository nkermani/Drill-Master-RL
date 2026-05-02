# src/warehouse_env/robot.py

"""Robot agent for warehouse environment"""

from dataclasses import dataclass
from typing import Optional, List
import numpy as np


@dataclass
class Robot:
    """Robot agent in the warehouse"""
    id: int
    position: np.ndarray
    state: str
    load: int = 0
    max_load: int = 1
    speed: float = 1.0
    battery: float = 100.0
    max_battery: float = 100.0
    current_task: Optional[int] = None
    path: List = None
    distance_traveled: float = 0.0

    def __post_init__(self):
        if self.path is None:
            self.path = []

    def reset(self, position: np.ndarray = None):
        """Reset robot to initial state"""
        if position is not None:
            self.position = position
        self.state = 'idle'
        self.load = 0
        self.battery = self.max_battery
        self.current_task = None
        self.path = []
        self.distance_traveled = 0.0

    def get_state_vector(self) -> np.ndarray:
        """Return 6-dim state vector"""
        state_map = {'idle': 0, 'moving': 1, 'loading': 2, 'unloading': 3, 'charging': 4}
        state_idx = state_map.get(self.state, 0)
        return np.array([
            self.position[0],
            self.position[1],
            state_idx,
            self.load,
            self.distance_traveled,
            0.0
        ], dtype=np.float32)

    def move_towards(self, target: np.ndarray, dt: float = 1.0) -> bool:
        """Move towards target position. Returns True if reached."""
        if np.array_equal(self.position, target):
            return True

        direction = target - self.position
        distance = np.linalg.norm(direction)
        if distance <= self.speed * dt:
            self.position = target.copy()
            return True

        self.position += (direction / distance) * self.speed * dt
        self.distance_traveled += self.speed * dt
        return False

    def can_pickup(self) -> bool:
        """Check if robot can pick up a task"""
        return self.state == 'idle' and self.load < self.max_load

    def can_deliver(self) -> bool:
        """Check if robot has a task to deliver"""
        return self.state == 'idle' and self.current_task is not None

    def pickup(self, task_id: int):
        """Pick up a task"""
        self.current_task = task_id
        self.load = 1
        self.state = 'moving'

    def deliver(self):
        """Deliver current task"""
        self.current_task = None
        self.load = 0
        self.state = 'idle'

    def update_battery(self, dt: float = 1.0):
        """Update battery level"""
        consumption = 0.1 * dt if self.state == 'moving' else 0.05 * dt
        self.battery = max(0, self.battery - consumption)
        if self.battery <= 20:
            self.state = 'charging'


__all__ = ['Robot']
