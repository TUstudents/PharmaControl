"""
Reinforcement Learning components for AutoPharm V3.

This module provides RL environments and training utilities
for learning control policies through environmental interaction.
"""

# Progressive imports as components become available
try:
    from .environment import GranulationEnv

    __all__ = ["GranulationEnv"]
except ImportError:
    __all__ = []
