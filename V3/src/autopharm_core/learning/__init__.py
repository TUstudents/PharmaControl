"""
Learning components for AutoPharm V3.

This module provides online learning, data handling, and model training
capabilities for continuous adaptation and improvement.
"""

# Progressive imports as components become available
try:
    from .data_handler import DataHandler

    __all__ = ["DataHandler"]
except ImportError:
    __all__ = []

try:
    from .online_trainer import OnlineTrainer

    __all__.append("OnlineTrainer")
except ImportError:
    pass
