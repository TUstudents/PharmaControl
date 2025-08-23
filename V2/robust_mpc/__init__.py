"""
RobustMPC Library - Industrial-Grade Model Predictive Control for Pharmaceutical Manufacturing

A comprehensive, production-ready library implementing advanced Model Predictive Control
algorithms with uncertainty quantification, bias correction, and robust optimization.
Specifically designed for pharmaceutical continuous manufacturing processes including
granulation, tableting, and coating operations.

Mathematical Framework:
    - State Estimation: Bias-augmented Kalman filtering with systematic error correction
    - Probabilistic Modeling: Transformer networks with Monte Carlo uncertainty quantification
    - Robust Optimization: Genetic algorithms for non-convex, constrained optimization
    - Risk-Aware Control: Multi-objective MPC balancing performance and robustness

Key Innovations:
    - Process vs Measurement bias correction paradigms for accurate state estimation
    - Uncertainty-aware control decisions using probabilistic predictions
    - Adaptive integral action for offset-free tracking despite unmeasured disturbances
    - Constraint-handling optimization for complex pharmaceutical process limits

Industrial Applications:
    - Continuous granulation: Particle size and moisture control
    - Tablet manufacturing: Weight uniformity and dissolution optimization
    - Coating processes: Thickness and quality attribute control
    - API synthesis: Yield and purity optimization under uncertainty

Version: 2.0.0
Author: PharmaControl-Pro Development Team
License: Educational/Research Use
Copyright: 2024 Advanced Process Control Research Group
"""

__version__ = "2.0.0"
__author__ = "PharmaControl-Pro Development Team"

# Lazy imports for PyTorch-dependent modules to avoid circular import issues
import importlib
import sys

from .data_buffer import DataBuffer, StartupHistoryGenerator

# Import key classes for easy access
from .estimators import (
    BiasAugmentedKalmanStateEstimator,
    KalmanStateEstimator,
    MeasurementBiasKalmanEstimator,
    ProcessBiasKalmanEstimator,
)

# Global variables to cache imported classes
_ProbabilisticTransformer = None
_GeneticOptimizer = None
_RobustMPCController = None


def _get_ProbabilisticTransformer():
    """Get ProbabilisticTransformer class with lazy loading."""
    global _ProbabilisticTransformer
    if _ProbabilisticTransformer is None:
        try:
            from .models import ProbabilisticTransformer

            _ProbabilisticTransformer = ProbabilisticTransformer
        except ImportError as e:
            # If there's still a circular import, try absolute import
            try:
                models_module = importlib.import_module("V2.robust_mpc.models")
                _ProbabilisticTransformer = models_module.ProbabilisticTransformer
            except ImportError:
                raise ImportError(f"Failed to import ProbabilisticTransformer: {e}")
    return _ProbabilisticTransformer


def _get_GeneticOptimizer():
    """Get GeneticOptimizer class with lazy loading."""
    global _GeneticOptimizer
    if _GeneticOptimizer is None:
        try:
            from .optimizers import GeneticOptimizer

            _GeneticOptimizer = GeneticOptimizer
        except ImportError as e:
            try:
                optimizers_module = importlib.import_module("V2.robust_mpc.optimizers")
                _GeneticOptimizer = optimizers_module.GeneticOptimizer
            except ImportError:
                raise ImportError(f"Failed to import GeneticOptimizer: {e}")
    return _GeneticOptimizer


def _get_RobustMPCController():
    """Get RobustMPCController class with lazy loading."""
    global _RobustMPCController
    if _RobustMPCController is None:
        try:
            from .core import RobustMPCController

            _RobustMPCController = RobustMPCController
        except ImportError as e:
            try:
                core_module = importlib.import_module("V2.robust_mpc.core")
                _RobustMPCController = core_module.RobustMPCController
            except ImportError:
                raise ImportError(f"Failed to import RobustMPCController: {e}")
    return _RobustMPCController


# V1 Integration Classes - Lazy loading
_V1ControllerAdapter = None
_V1_MPC_Wrapper = None


def _get_V1ControllerAdapter():
    """Get V1ControllerAdapter class with lazy loading."""
    global _V1ControllerAdapter
    if _V1ControllerAdapter is None:
        try:
            from .v1_adapter import V1ControllerAdapter

            _V1ControllerAdapter = V1ControllerAdapter
        except ImportError as e:
            try:
                v1_adapter_module = importlib.import_module("V2.robust_mpc.v1_adapter")
                _V1ControllerAdapter = v1_adapter_module.V1ControllerAdapter
            except ImportError:
                raise ImportError(f"Failed to import V1ControllerAdapter: {e}")
    return _V1ControllerAdapter


def _get_V1_MPC_Wrapper():
    """Get V1_MPC_Wrapper class with lazy loading."""
    global _V1_MPC_Wrapper
    if _V1_MPC_Wrapper is None:
        try:
            from .v1_adapter import V1_MPC_Wrapper

            _V1_MPC_Wrapper = V1_MPC_Wrapper
        except ImportError as e:
            try:
                v1_adapter_module = importlib.import_module("V2.robust_mpc.v1_adapter")
                _V1_MPC_Wrapper = v1_adapter_module.V1_MPC_Wrapper
            except ImportError:
                raise ImportError(f"Failed to import V1_MPC_Wrapper: {e}")
    return _V1_MPC_Wrapper


# Create properties that lazily load the classes
def __getattr__(name):
    """Module-level __getattr__ for lazy loading."""
    if name == "ProbabilisticTransformer":
        return _get_ProbabilisticTransformer()
    elif name == "GeneticOptimizer":
        return _get_GeneticOptimizer()
    elif name == "RobustMPCController":
        return _get_RobustMPCController()
    elif name == "V1ControllerAdapter":
        return _get_V1ControllerAdapter()
    elif name == "V1_MPC_Wrapper":
        return _get_V1_MPC_Wrapper()
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Define what gets imported with "from robust_mpc import *"
__all__ = [
    # State Estimation
    "KalmanStateEstimator",
    "BiasAugmentedKalmanStateEstimator",
    "ProcessBiasKalmanEstimator",
    "MeasurementBiasKalmanEstimator",
    # Data Management
    "DataBuffer",
    "StartupHistoryGenerator",
    # Probabilistic Modeling
    "ProbabilisticTransformer",
    # Optimization
    "GeneticOptimizer",
    # Control
    "RobustMPCController",
    # V1 Integration
    "V1ControllerAdapter",
    "V1_MPC_Wrapper",
]

# Library metadata
LIBRARY_INFO = {
    "name": "RobustMPC",
    "version": __version__,
    "description": "Industrial-grade Model Predictive Control with uncertainty quantification",
    "components": {
        "estimators": "State estimation algorithms (Kalman Filter, EKF, UKF)",
        "models": "Probabilistic predictive models with uncertainty quantification",
        "optimizers": "Advanced optimization algorithms (GA, PSO, Bayesian)",
        "core": "Main MPC controller classes with robustness guarantees",
    },
    "target_applications": [
        "Pharmaceutical continuous manufacturing",
        "Chemical process control",
        "Advanced materials processing",
        "Food and beverage production",
    ],
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
    for component, description in info["components"].items():
        print(f"  • {component:12}: {description}")

    print(f"\nTarget Applications:")
    for app in info["target_applications"]:
        print(f"  • {app}")
    print(f"{'='*60}\n")


# Validate that all expected modules can be imported
def _validate_imports():
    """Validate that all library components can be imported."""
    try:
        from . import core, estimators, models, optimizers

        return True
    except ImportError as e:
        print(f"Warning: Some RobustMPC components not available: {e}")
        return False


# Configuration defaults
DEFAULT_CONFIG = {
    "estimation": {
        "kalman_process_noise": 1.0,
        "kalman_measurement_noise": 10.0,
        "initial_covariance_scale": 1.0,
    },
    "modeling": {"uncertainty_samples": 50, "dropout_rate": 0.1, "confidence_level": 0.95},
    "optimization": {
        "population_size": 50,
        "generations": 20,
        "mutation_rate": 0.1,
        "crossover_rate": 0.7,
    },
    "control": {
        "prediction_horizon": 72,
        "control_horizon": 36,
        "integral_gain": 0.1,
        "control_effort_weight": 0.05,
        "verbose": False,  # Production-ready default
    },
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
