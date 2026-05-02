# src/warehouse_env/task.py

"""Task generation and management for warehouse environment"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, List
import numpy as np


class TaskStatus(Enum):
    PENDING = 0
    ASSIGNED = 1
    IN_PROGRESS = 2
    COMPLETED = 3
    CANCELLED = 4


@dataclass
class Task:
    """Pick-up and delivery task"""
    id: int
    pickup_location: np.ndarray
    delivery_location: np.ndarray
    creation_time: float
    status: TaskStatus = TaskStatus.PENDING
    assigned_robot: Optional[int] = None
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    priority: int = 1
    weight: float = 1.0

    def assign(self, robot_id: int, current_time: float):
        """Assign task to a robot"""
        self.status = TaskStatus.ASSIGNED
        self.assigned_robot = robot_id
        self.start_time = current_time

    def start(self):
        """Mark task as in progress"""
        self.status = TaskStatus.IN_PROGRESS

    def complete(self, current_time: float):
        """Mark task as completed"""
        self.status = TaskStatus.COMPLETED
        self.completion_time = current_time

    def get_features(self) -> np.ndarray:
        """Return 7-dim task features"""
        return np.array([
            self.pickup_location[0],
            self.pickup_location[1],
            self.delivery_location[0],
            self.delivery_location[1],
            self.priority,
            self.weight,
            self.creation_time
        ], dtype=np.float32)

    @property
    def is_active(self) -> bool:
        """Check if task is still active"""
        return self.status in [TaskStatus.PENDING, TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS]

    @property
    def wait_time(self, current_time: float) -> float:
        """Calculate wait time"""
        return current_time - self.creation_time


def generate_task(task_id: int, current_time: float, grid_size: tuple,
                 num_stations: int, stations: list) -> Task:
    """Generate a random task"""
    pickup_idx = np.random.randint(0, num_stations)
    delivery_idx = np.random.randint(0, num_stations)

    while delivery_idx == pickup_idx:
        delivery_idx = np.random.randint(0, num_stations)

    pickup_location = np.array(stations[pickup_idx], dtype=np.float32)
    delivery_location = np.array(stations[delivery_idx], dtype=np.float32)

    return Task(
        id=task_id,
        pickup_location=pickup_location,
        delivery_location=delivery_location,
        creation_time=current_time,
        priority=np.random.randint(1, 4),
        weight=np.random.uniform(0.5, 2.0)
    )


__all__ = ['Task', 'TaskStatus', 'generate_task']
