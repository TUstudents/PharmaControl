"""PharmaControl V1: Prototype Transformer-based MPC for Continuous Granulation.

This package implements a prototype Model Predictive Control system for pharmaceutical
continuous granulation processes using transformer-based neural network predictions.

The V1 implementation serves as an educational and research prototype, demonstrating
the integration of modern deep learning techniques with classical process control
for pharmaceutical manufacturing applications.

Modules:
    plant_simulator: High-fidelity granulation process simulator with realistic
        dynamics, time delays, and disturbances for testing control algorithms
    model_architecture: Transformer encoder-decoder architecture for multi-step
        process prediction with positional encoding and attention mechanisms
    dataset: PyTorch Dataset implementation for time series sequence extraction
        from continuous process data with sliding window approach
    mpc_controller: Discrete optimization-based Model Predictive Controller using
        exhaustive grid search over transformer predictions

Key Features:
    - Transformer-based process prediction with attention mechanisms
    - Discrete MPC optimization with constraint handling
    - Realistic process simulation including time delays and disturbances
    - Educational notebooks demonstrating complete control system development
    - Integration with pharmaceutical process terminology (CMAs, CPPs)

Process Variables:
    Critical Material Attributes (CMAs):
        - d50: Median particle size distribution (micrometers)
        - lod: Loss on drying/moisture content (percentage)

    Critical Process Parameters (CPPs):
        - spray_rate: Liquid binder spray rate (g/min)
        - air_flow: Fluidization air flow rate (m�/h)
        - carousel_speed: Carousel rotation speed (rpm)
        - specific_energy: Calculated soft sensor (spray_rate � carousel_speed / 1000)
        - froude_number_proxy: Calculated soft sensor (carousel_speed� / 9.81)

Typical Usage:
    >>> from V1.src import plant_simulator, model_architecture, mpc_controller, dataset
    >>>
    >>> # Create process simulator
    >>> plant = plant_simulator.AdvancedPlantSimulator()
    >>>
    >>> # Load trained model and create MPC controller
    >>> model = torch.load('best_predictor_model.pth')
    >>> controller = mpc_controller.MPCController(model, config, constraints, scalers)
    >>>
    >>> # Run closed-loop control
    >>> cpps = {'spray_rate': 120.0, 'air_flow': 500.0, 'carousel_speed': 30.0}
    >>> state = plant.step(cpps)
    >>> optimal_action = controller.suggest_action(past_cmas, past_cpps, targets)

Architecture Limitations (Addressed in V2):
    - Exhaustive grid search: O(n^k) computational complexity
    - No uncertainty quantification: Single-point predictions
    - Basic constraint handling: Hard filtering only
    - No integral action: Susceptible to steady-state offset
    - Limited scalability: Discrete optimization approach

Educational Value:
    This prototype implementation prioritizes clarity and understanding over
    performance, making it ideal for:
    - Learning MPC fundamentals with modern ML predictions
    - Understanding transformer applications in process control
    - Developing pharmaceutical process control intuition
    - Benchmarking against advanced V2 robust implementation

Version: 1.0.0
License: See LICENSE file in project root
Authors: PharmaControl Development Team
"""

# Import key classes for convenient access
from .plant_simulator import AdvancedPlantSimulator
from .model_architecture import GranulationPredictor, PositionalEncoding
from .mpc_controller import MPCController
from .dataset import GranulationDataset

# Package metadata
__version__ = "1.0.0"
__author__ = "PharmaControl Development Team"
__email__ = "contact@pharmacontrol.ai"
__description__ = "Prototype Transformer-based MPC for Pharmaceutical Continuous Granulation"

# Define public API
__all__ = [
    # Core classes
    "AdvancedPlantSimulator",
    "GranulationPredictor",
    "PositionalEncoding",
    "MPCController",
    "GranulationDataset",
    # Package metadata
    "__version__",
    "__author__",
    "__email__",
    "__description__",
]

# Version compatibility check
import sys

if sys.version_info < (3, 9):
    raise ImportError("PharmaControl V1 requires Python 3.9 or higher")

# Optional dependency checks with informative error messages
try:
    import torch

    if torch.__version__ < "1.9.0":
        import warnings

        warnings.warn(
            f"PyTorch version {torch.__version__} detected. "
            "PharmaControl V1 is tested with PyTorch >= 1.9.0. "
            "Some features may not work correctly with older versions.",
            UserWarning,
        )
except ImportError:
    raise ImportError(
        "PyTorch is required for PharmaControl V1. " "Install with: pip install torch>=1.9.0"
    )

try:
    import pandas as pd

    if pd.__version__ < "1.3.0":
        import warnings

        warnings.warn(
            f"Pandas version {pd.__version__} detected. "
            "PharmaControl V1 is tested with Pandas >= 1.3.0.",
            UserWarning,
        )
except ImportError:
    raise ImportError(
        "Pandas is required for PharmaControl V1. " "Install with: pip install pandas>=1.3.0"
    )

try:
    import numpy as np

    if np.__version__ < "1.20.0":
        import warnings

        warnings.warn(
            f"NumPy version {np.__version__} detected. "
            "PharmaControl V1 is tested with NumPy >= 1.20.0.",
            UserWarning,
        )
except ImportError:
    raise ImportError(
        "NumPy is required for PharmaControl V1. " "Install with: pip install numpy>=1.20.0"
    )
