"""
AutoPharm V3: Autonomous Pharmaceutical Process Control Framework

A next-generation control system that combines:
- Online Learning & Adaptation
- Explainable AI & Trust  
- Advanced Policy Learning (Reinforcement Learning)

Version: 3.0.0
"""

__version__ = "3.0.0"
__author__ = "PharmaControl Development Team"

# Progressive imports as components become available
try:
    from .common.types import StateVector, ControlAction, ModelPrediction, TrainingMetrics, DecisionExplanation
except ImportError:
    pass

try:
    from .learning.data_handler import DataHandler
    from .learning.online_trainer import OnlineTrainer
except ImportError:
    pass

try:
    from .xai.explainer import ShapExplainer
except ImportError:
    pass

try:
    from .rl.environment import GranulationEnv
except ImportError:
    pass

def get_version():
    """Get the current version of AutoPharm."""
    return __version__

def print_library_info():
    """Print information about the AutoPharm library."""
    print(f"AutoPharm V3 Framework - Version {__version__}")
    print("Autonomous Pharmaceutical Process Control")
    print("Core Pillars:")
    print("  1. Online Learning & Adaptation ✓")
    print("  2. Explainable AI & Trust ✓") 
    print("  3. Advanced Policy Learning (RL) ✓")
    print("\nAll three pillars now implemented!")