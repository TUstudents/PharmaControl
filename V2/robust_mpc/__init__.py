"""
RobustMPC Library - Industrial-Grade Model Predictive Control

A comprehensive library for uncertainty-aware, adaptive control systems 
designed for pharmaceutical and chemical process applications.

Key Components:
- State Estimation: Kalman filtering for noisy sensor data
- Probabilistic Models: Uncertainty-aware predictive models  
- Advanced Optimization: Genetic algorithms and multi-objective optimization
- Robust Controllers: Offset-free MPC with constraint guarantees

Version: 2.0.0
Author: PharmaControl-Pro Development Team
License: Educational/Research Use
"""

__version__ = "2.0.0"
__author__ = "PharmaControl-Pro Development Team"

# Import key classes for easy access
from .estimators import KalmanStateEstimator
from .models import ProbabilisticTransformer  # ✅ Available as of V2-2
from .optimizers import GeneticOptimizer      # ✅ Available as of V2-3
from .core import RobustMPCController         # ✅ Available as of V2-4

# Define what gets imported with "from robust_mpc import *"
__all__ = [
    'KalmanStateEstimator',
    'ProbabilisticTransformer', 
    'GeneticOptimizer',        # ✅ Available as of V2-3
    'RobustMPCController'      # ✅ Available as of V2-4
]

# Library metadata
LIBRARY_INFO = {
    'name': 'RobustMPC',
    'version': __version__,
    'description': 'Industrial-grade Model Predictive Control with uncertainty quantification',
    'components': {
        'estimators': 'State estimation algorithms (Kalman Filter, EKF, UKF)',
        'models': 'Probabilistic predictive models with uncertainty quantification',
        'optimizers': 'Advanced optimization algorithms (GA, PSO, Bayesian)',
        'core': 'Main MPC controller classes with robustness guarantees'
    },
    'target_applications': [
        'Pharmaceutical continuous manufacturing',
        'Chemical process control',
        'Advanced materials processing',
        'Food and beverage production'
    ]
}

def get_library_info():
    """Return comprehensive library information."""
    return LIBRARY_INFO

def print_library_info():
    """Print formatted library information."""
    info = LIBRARY_INFO
    print(f"\n{'='*60}")
    print(f"  {info['name']} Library v{info['version']}")
    print(f"{'='*60}")
    print(f"Description: {info['description']}")
    print(f"\nComponents:")
    for component, description in info['components'].items():
        print(f"  • {component:12}: {description}")
    
    print(f"\nTarget Applications:")
    for app in info['target_applications']:
        print(f"  • {app}")
    print(f"{'='*60}\n")

# Validate that all expected modules can be imported
def _validate_imports():
    """Validate that all library components can be imported."""
    try:
        from . import estimators
        from . import models  
        from . import optimizers
        from . import core
        return True
    except ImportError as e:
        print(f"Warning: Some RobustMPC components not available: {e}")
        return False

# Configuration defaults
DEFAULT_CONFIG = {
    'estimation': {
        'kalman_process_noise': 1.0,
        'kalman_measurement_noise': 10.0,
        'initial_covariance_scale': 1.0
    },
    'modeling': {
        'uncertainty_samples': 50,
        'dropout_rate': 0.1,
        'confidence_level': 0.95
    },
    'optimization': {
        'population_size': 50,
        'generations': 20,
        'mutation_rate': 0.1,
        'crossover_rate': 0.7
    },
    'control': {
        'prediction_horizon': 72,
        'control_horizon': 36,
        'integral_gain': 0.1,
        'control_effort_weight': 0.05
    }
}

def get_default_config():
    """Return default configuration dictionary."""
    return DEFAULT_CONFIG.copy()

# Version compatibility check
MIN_PYTHON_VERSION = (3, 9)  # Minimum required Python version
import sys

if sys.version_info < MIN_PYTHON_VERSION:
    raise RuntimeError(
        f"RobustMPC requires Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]} or higher. "
        f"Current version: {sys.version_info.major}.{sys.version_info.minor}"
    )