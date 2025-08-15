====================
API Reference
====================

This section provides a complete API reference for all classes, functions, and modules 
in the PharmaControl V1 package.

.. toctree::
   :maxdepth: 2

   plant_simulator
   model_architecture
   mpc_controller
   dataset

Module Overview
===============

The PharmaControl V1 API is organized into four main modules:

Plant Simulation
----------------

.. currentmodule:: plant_simulator

.. autosummary::
   :toctree: generated/

   AdvancedPlantSimulator

The plant simulator provides high-fidelity modeling of pharmaceutical continuous granulation 
processes with realistic dynamics, time delays, and disturbances.

Model Architecture
------------------

.. currentmodule:: model_architecture

.. autosummary::
   :toctree: generated/

   PositionalEncoding
   GranulationPredictor

The model architecture module implements transformer-based neural networks specifically 
designed for pharmaceutical process prediction and MPC applications.

MPC Controller
--------------

.. currentmodule:: mpc_controller

.. autosummary::
   :toctree: generated/

   MPCController

The MPC controller implements discrete optimization-based Model Predictive Control using 
transformer predictions and constraint handling.

Dataset
-------

.. currentmodule:: dataset

.. autosummary::
   :toctree: generated/

   GranulationDataset

The dataset module provides efficient PyTorch Dataset implementation for pharmaceutical 
time series data with sliding window sequence extraction.

Quick Reference
===============

Core Classes
------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - :class:`~plant_simulator.AdvancedPlantSimulator`
     - High-fidelity granulation process simulator
   * - :class:`~model_architecture.GranulationPredictor`
     - Transformer-based process prediction model
   * - :class:`~mpc_controller.MPCController`
     - Model Predictive Controller with discrete optimization
   * - :class:`~dataset.GranulationDataset`
     - PyTorch Dataset for time series sequence extraction

Key Methods
-----------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Method
     - Description
   * - :meth:`~plant_simulator.AdvancedPlantSimulator.step`
     - Execute one simulation time step
   * - :meth:`~model_architecture.GranulationPredictor.forward`
     - Forward pass for process prediction
   * - :meth:`~mpc_controller.MPCController.suggest_action`
     - Compute optimal control action using MPC
   * - :meth:`~dataset.GranulationDataset.__getitem__`
     - Extract training sequence from dataset

Data Structures
===============

Process Variables
-----------------

**Critical Material Attributes (CMAs)**

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Variable
     - Units
     - Description
   * - d50
     - μm
     - Median particle size distribution
   * - lod
     - %
     - Loss on drying (moisture content)

**Critical Process Parameters (CPPs)**

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Variable
     - Units
     - Description
   * - spray_rate
     - g/min
     - Liquid binder spray rate
   * - air_flow
     - m³/h
     - Fluidization air flow rate
   * - carousel_speed
     - rpm
     - Carousel rotation speed
   * - specific_energy
     - -
     - Calculated: spray_rate × carousel_speed / 1000
   * - froude_number_proxy
     - -
     - Calculated: carousel_speed² / 9.81

Configuration Structures
-------------------------

**MPC Configuration**

.. code-block:: python

   config = {
       'cpp_names': ['spray_rate', 'air_flow', 'carousel_speed'],
       'cma_names': ['d50', 'lod'],
       'cpp_names_and_soft_sensors': [
           'spray_rate', 'air_flow', 'carousel_speed', 
           'specific_energy', 'froude_number_proxy'
       ],
       'horizon': 10,                    # Prediction horizon
       'discretization_steps': 5,        # Grid discretization
       'control_effort_lambda': 0.1      # Control effort weight
   }

**Constraint Configuration**

.. code-block:: python

   constraints = {
       'spray_rate': {
           'min_val': 80.0,              # Minimum value
           'max_val': 180.0,             # Maximum value  
           'max_change_per_step': 10.0   # Maximum change per step
       },
       # ... similar for other variables
   }

Tensor Shapes
=============

**Model Input/Output Shapes**

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Tensor
     - Shape
     - Description
   * - past_cmas
     - (batch, lookback, 2)
     - Historical CMA observations
   * - past_cpps
     - (batch, lookback, 5)
     - Historical CPP measurements
   * - future_cpps
     - (batch, horizon, 5)
     - Planned control sequence
   * - predictions
     - (batch, horizon, 2)
     - Predicted CMA values

**Dataset Output Shapes**

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Component
     - Shape
     - Description
   * - past_cmas
     - (lookback, 2)
     - Historical material attributes
   * - past_cpps
     - (lookback, 5)
     - Historical process parameters
   * - future_cpps
     - (horizon, 5)
     - Future control actions
   * - targets
     - (horizon, 2)
     - Ground truth CMA targets

Error Handling
==============

Common Exceptions
-----------------

.. py:exception:: ValueError

   Raised when:
   * Configuration parameters are missing or invalid
   * Array shapes are incompatible
   * Control constraints are violated
   * Required variables not found in data

.. py:exception:: RuntimeError

   Raised when:
   * Model prediction fails
   * GPU memory allocation errors
   * Optimization timeout exceeded

.. py:exception:: IndexError

   Raised when:
   * Sequence indices exceed data bounds
   * Array indexing errors in dataset

Exception Handling Examples
---------------------------

.. code-block:: python

   try:
       controller = MPCController(model, config, constraints, scalers)
       optimal_action = controller.suggest_action(past_cmas, past_cpps, targets)
   except ValueError as e:
       print(f"Configuration error: {e}")
   except RuntimeError as e:
       print(f"Runtime error: {e}")
   except Exception as e:
       print(f"Unexpected error: {e}")

Performance Guidelines
======================

Computational Complexity
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Component
     - Complexity
     - Notes
   * - MPC Optimization
     - O(n^k)
     - n=discretization, k=variables
   * - Transformer Forward
     - O(L²d)
     - L=sequence length, d=model dim
   * - Dataset Loading
     - O(N)
     - N=dataset size

Memory Requirements
-------------------

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Component
     - Memory Usage
     - Notes
   * - Model Parameters
     - 1-5 MB
     - Depends on architecture
   * - Training Batch
     - 50-200 MB
     - Batch size dependent
   * - Full Dataset
     - 100-1000 MB
     - Dataset size dependent

Optimization Tips
-----------------

1. **Reduce discretization steps** for faster MPC optimization
2. **Use smaller batch sizes** if encountering memory issues  
3. **Enable GPU acceleration** for transformer inference
4. **Use DataLoader prefetching** for faster training
5. **Consider model quantization** for deployment

Version Compatibility
=====================

**Python Requirements**

* Python ≥ 3.9
* PyTorch ≥ 1.9.0
* NumPy ≥ 1.20.0
* Pandas ≥ 1.3.0

**Optional Dependencies**

* Matplotlib ≥ 3.3.0 (for visualization)
* Seaborn ≥ 0.11.0 (for advanced plotting)
* Jupyter ≥ 1.0.0 (for educational notebooks)
* scikit-learn ≥ 0.24.0 (for preprocessing)

Migration Guide
===============

From V1 to V2
--------------

Key changes when upgrading to V2:

1. **Optimization**: Replace exhaustive search with genetic algorithms
2. **Uncertainty**: Add probabilistic predictions with Monte Carlo dropout
3. **State Estimation**: Include Kalman filtering for sensor noise
4. **Constraints**: Use soft constraints with penalty functions

.. code-block:: python

   # V1 approach
   controller_v1 = MPCController(model, config, constraints, scalers)
   
   # V2 approach  
   from robust_mpc import RobustMPCController, KalmanStateEstimator, GeneticOptimizer
   
   estimator = KalmanStateEstimator(...)
   optimizer = GeneticOptimizer(...)
   controller_v2 = RobustMPCController(model, estimator, optimizer, config, scalers)

See Also
========

* :doc:`../tutorials/index` - Step-by-step tutorials
* :doc:`../examples/index` - Practical usage examples  
* :doc:`../modules/plant_simulator` - Detailed plant simulator documentation
* :doc:`../modules/model_architecture` - Transformer architecture details
* :doc:`../modules/mpc_controller` - MPC implementation guide
* :doc:`../modules/dataset` - Dataset usage and optimization