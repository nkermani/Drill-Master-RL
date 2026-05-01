# src/train/ppo_agent/methods/__init__.py

from .compute_returns import compute_returns
from .update import update

__all__ = ['compute_returns', 'update']
