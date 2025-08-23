"""
Common utilities and types for AutoPharm V3.

This module provides shared data structures and utility functions
used across all components of the AutoPharm framework.
"""

# Progressive imports as components become available
try:
    from .types import (
        StateVector,
        ControlAction,
        ModelPrediction,
        TrainingMetrics,
        DecisionExplanation,
    )

    __all__ = [
        "StateVector",
        "ControlAction",
        "ModelPrediction",
        "TrainingMetrics",
        "DecisionExplanation",
    ]
except ImportError:
    __all__ = []
