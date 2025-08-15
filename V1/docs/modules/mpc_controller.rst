====================
MPC Controller
====================

.. currentmodule:: mpc_controller

The MPC controller module implements a discrete optimization-based Model Predictive Controller 
that uses transformer-based neural network predictions to find optimal control actions while 
respecting process constraints.

Overview
========

The :class:`MPCController` implements a classic MPC architecture with modern neural network 
prediction models. It performs exhaustive grid search over discretized control actions to 
find the optimal control sequence that minimizes a weighted combination of tracking error 
and control effort.

**Key Features:**

* **Discrete optimization** with exhaustive grid search
* **Constraint handling** through candidate filtering
* **Multi-objective cost function** balancing tracking and control effort
* **Transformer integration** for multi-step process prediction
* **Robust error handling** with graceful degradation

MPC Algorithm
=============

The controller implements the standard MPC optimization problem:

.. math::

   \min_{u} \sum_{k=0}^{H-1} \left[ \|y_{k+1|t} - r_{k+1}\|_1 + \lambda \|u_{k|t} - u_{t-1}\|_1 \right]

Subject to:

.. math::

   u_{\min} \leq u_{k|t} \leq u_{\max}, \quad k = 0, \ldots, H-1

Where:

* :math:`y_{k+1|t}` = predicted CMA at time t+k+1 given information at time t
* :math:`r_{k+1}` = reference setpoint for CMAs
* :math:`u_{k|t}` = control action at time t+k given information at time t  
* :math:`u_{t-1}` = previous control action (for move suppression)
* :math:`H` = prediction horizon
* :math:`\lambda` = control effort weighting factor

Optimization Strategy
====================

The V1 implementation uses exhaustive grid search:

1. **Candidate Generation**: Create Cartesian product of discretized control changes
2. **Constraint Filtering**: Remove candidates violating operational limits
3. **Batch Evaluation**: Use neural network to predict process response
4. **Cost Calculation**: Evaluate weighted cost function for each candidate
5. **Selection**: Choose candidate with minimum cost

**Computational Complexity**: O(n^k) where n = discretization steps, k = control variables

**Performance Characteristics**:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Variables
     - Discretization
     - Candidates
     - Real-time?
   * - 3
     - 5
     - 125
     - ✅ Yes
   * - 3
     - 10
     - 1,000
     - ⚠️ Marginal
   * - 5
     - 10
     - 100,000
     - ❌ No

Class Reference
===============

.. autoclass:: MPCController
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
=============

The MPC controller requires several configuration dictionaries:

**Model Configuration (config)**

.. code-block:: python

   config = {
       'cpp_names': ['spray_rate', 'air_flow', 'carousel_speed'],
       'cma_names': ['d50', 'lod'],
       'cpp_names_and_soft_sensors': [
           'spray_rate', 'air_flow', 'carousel_speed', 
           'specific_energy', 'froude_number_proxy'
       ],
       'horizon': 10,                    # Prediction horizon steps
       'discretization_steps': 5,        # Grid discretization per variable
       'control_effort_lambda': 0.1      # Control effort penalty weight
   }

**Constraint Configuration (constraints)**

.. code-block:: python

   constraints = {
       'spray_rate': {
           'min_val': 80.0,              # Minimum spray rate (g/min)
           'max_val': 180.0,             # Maximum spray rate (g/min)
           'max_change_per_step': 10.0   # Maximum change per step (g/min)
       },
       'air_flow': {
           'min_val': 400.0,             # Minimum air flow (m³/h)
           'max_val': 700.0,             # Maximum air flow (m³/h)
           'max_change_per_step': 50.0   # Maximum change per step (m³/h)
       },
       'carousel_speed': {
           'min_val': 20.0,              # Minimum carousel speed (rpm)
           'max_val': 40.0,              # Maximum carousel speed (rpm)
           'max_change_per_step': 5.0    # Maximum change per step (rpm)
       }
   }

**Scalers Dictionary**

.. code-block:: python

   import joblib
   
   # Load fitted scalers from training
   scalers = joblib.load('data/scalers.joblib')
   
   # Scalers should contain entries for all variables:
   # scalers['spray_rate'], scalers['air_flow'], scalers['carousel_speed']
   # scalers['specific_energy'], scalers['froude_number_proxy']
   # scalers['d50'], scalers['lod']

Usage Examples
==============

Basic MPC Setup
---------------

.. code-block:: python

   import torch
   import joblib
   import pandas as pd
   import numpy as np
   from V1.src.mpc_controller import MPCController
   
   # Load trained model and scalers
   model = torch.load('data/best_predictor_model.pth')
   scalers = joblib.load('data/scalers.joblib')
   
   # Configuration
   config = {
       'cpp_names': ['spray_rate', 'air_flow', 'carousel_speed'],
       'cma_names': ['d50', 'lod'],
       'cpp_names_and_soft_sensors': [
           'spray_rate', 'air_flow', 'carousel_speed', 
           'specific_energy', 'froude_number_proxy'
       ],
       'horizon': 10,
       'discretization_steps': 5,
       'control_effort_lambda': 0.1
   }
   
   constraints = {
       'spray_rate': {'min_val': 80.0, 'max_val': 180.0, 'max_change_per_step': 10.0},
       'air_flow': {'min_val': 400.0, 'max_val': 700.0, 'max_change_per_step': 50.0},
       'carousel_speed': {'min_val': 20.0, 'max_val': 40.0, 'max_change_per_step': 5.0}
   }
   
   # Create controller
   controller = MPCController(model, config, constraints, scalers)

Single Control Step
-------------------

.. code-block:: python

   # Prepare historical data (lookback window)
   # These should be DataFrames with the correct column names
   past_cmas = pd.DataFrame({
       'd50': [420.0, 415.0, 410.0, 405.0, 400.0],  # Recent particle sizes
       'lod': [1.8, 1.7, 1.6, 1.5, 1.4]             # Recent moisture levels  
   })
   
   past_cpps = pd.DataFrame({
       'spray_rate': [120.0, 118.0, 115.0, 112.0, 110.0],
       'air_flow': [500.0, 502.0, 505.0, 508.0, 510.0],
       'carousel_speed': [30.0, 30.5, 31.0, 31.5, 32.0],
       'specific_energy': [3.6, 3.59, 3.565, 3.528, 3.52],  # Calculated soft sensors
       'froude_number_proxy': [91.7, 94.8, 97.9, 101.0, 104.2]
   })
   
   # Define target setpoints (repeated over horizon)
   target_cmas = np.array([
       [380.0, 1.8],  # Target: d50=380μm, lod=1.8%
       [380.0, 1.8],
       [380.0, 1.8],
       [380.0, 1.8],
       [380.0, 1.8],
       [380.0, 1.8],
       [380.0, 1.8],
       [380.0, 1.8],
       [380.0, 1.8],
       [380.0, 1.8]
   ])
   
   # Get optimal control action
   optimal_action = controller.suggest_action(past_cmas, past_cpps, target_cmas)
   
   print(f"Optimal spray rate: {optimal_action[0]:.1f} g/min")
   print(f"Optimal air flow: {optimal_action[1]:.1f} m³/h")
   print(f"Optimal carousel speed: {optimal_action[2]:.1f} rpm")

Closed-loop Simulation
----------------------

.. code-block:: python

   from V1.src.plant_simulator import AdvancedPlantSimulator
   import matplotlib.pyplot as plt
   
   # Setup
   plant = AdvancedPlantSimulator()
   controller = MPCController(model, config, constraints, scalers)
   
   # Simulation parameters
   n_steps = 100
   lookback = 36  # Must match model training
   
   # Data storage
   time_data = []
   d50_data = []
   lod_data = []
   spray_rate_data = []
   air_flow_data = []
   carousel_speed_data = []
   
   # Initialize with steady-state data
   steady_state_cmas = pd.DataFrame({
       'd50': [400.0] * lookback,
       'lod': [1.5] * lookback
   })
   
   steady_state_cpps = pd.DataFrame({
       'spray_rate': [120.0] * lookback,
       'air_flow': [500.0] * lookback, 
       'carousel_speed': [30.0] * lookback,
       'specific_energy': [3.6] * lookback,
       'froude_number_proxy': [91.7] * lookback
   })
   
   # Target setpoint
   target = np.tile([380.0, 1.8], (10, 1))  # Target over horizon
   
   # Simulation loop
   for t in range(n_steps):
       # Get optimal control action
       optimal_action = controller.suggest_action(
           steady_state_cmas.tail(lookback), 
           steady_state_cpps.tail(lookback),
           target
       )
       
       # Apply to plant
       cpps = {
           'spray_rate': optimal_action[0],
           'air_flow': optimal_action[1], 
           'carousel_speed': optimal_action[2]
       }
       
       state = plant.step(cpps)
       
       # Calculate soft sensors
       specific_energy = (cpps['spray_rate'] * cpps['carousel_speed']) / 1000.0
       froude_number = (cpps['carousel_speed'] ** 2) / 9.81
       
       # Update data histories
       new_cma_row = pd.DataFrame({
           'd50': [state['d50']],
           'lod': [state['lod']]
       })
       
       new_cpp_row = pd.DataFrame({
           'spray_rate': [cpps['spray_rate']],
           'air_flow': [cpps['air_flow']],
           'carousel_speed': [cpps['carousel_speed']],
           'specific_energy': [specific_energy],
           'froude_number_proxy': [froude_number]
       })
       
       steady_state_cmas = pd.concat([steady_state_cmas, new_cma_row], ignore_index=True)
       steady_state_cpps = pd.concat([steady_state_cpps, new_cpp_row], ignore_index=True)
       
       # Store for plotting
       time_data.append(t)
       d50_data.append(state['d50'])
       lod_data.append(state['lod'])
       spray_rate_data.append(cpps['spray_rate'])
       air_flow_data.append(cpps['air_flow'])
       carousel_speed_data.append(cpps['carousel_speed'])
   
   # Plot results
   fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
   
   ax1.plot(time_data, d50_data, 'b-', label='Actual')
   ax1.axhline(y=380.0, color='r', linestyle='--', label='Target')
   ax1.set_ylabel('Particle Size (μm)')
   ax1.set_title('d50 Control Performance')
   ax1.legend()
   ax1.grid(True)
   
   ax2.plot(time_data, lod_data, 'g-', label='Actual')
   ax2.axhline(y=1.8, color='r', linestyle='--', label='Target')
   ax2.set_ylabel('Moisture Content (%)')
   ax2.set_title('LOD Control Performance')
   ax2.legend()
   ax2.grid(True)
   
   ax3.plot(time_data, spray_rate_data, 'c-')
   ax3.set_ylabel('Spray Rate (g/min)')
   ax3.set_title('Control Actions')
   ax3.grid(True)
   
   ax4.plot(time_data, air_flow_data, 'm-', label='Air Flow')
   ax4.plot(time_data, carousel_speed_data, 'y-', label='Carousel Speed')
   ax4.set_ylabel('Air Flow (m³/h) / Speed (rpm)')
   ax4.set_xlabel('Time (steps)')
   ax4.legend()
   ax4.grid(True)
   
   plt.tight_layout()
   plt.show()

Tuning Guidelines
=================

**Horizon Length**

* **Short horizon** (5-10 steps): Faster computation, more reactive control
* **Long horizon** (15-25 steps): Better anticipation, potential stability issues
* **Rule of thumb**: 2-3× process settling time

**Discretization Steps**

* **Coarse discretization** (3-5 steps): Fast computation, suboptimal solutions
* **Fine discretization** (7-12 steps): Better optimality, exponential complexity growth
* **Adaptive**: Start coarse, refine around promising regions

**Control Effort Weight (λ)**

* **Low weight** (0.01-0.05): Aggressive control, faster response
* **High weight** (0.2-0.5): Smooth control, slower response
* **Tuning**: Start with 0.1, adjust based on control performance

**Constraint Tuning**

* **Max change limits**: Balance responsiveness vs. equipment wear
* **Safety margins**: Account for model uncertainty and measurement noise
* **Operational limits**: Conservative bounds for safe operation

Performance Analysis
====================

**Computational Profiling**

.. code-block:: python

   import time
   import cProfile
   
   def profile_mpc_step():
       # Setup (use real data)
       controller = MPCController(model, config, constraints, scalers)
       
       # Time single optimization
       start_time = time.time()
       optimal_action = controller.suggest_action(past_cmas, past_cpps, target_cmas)
       end_time = time.time()
       
       print(f"MPC optimization time: {(end_time - start_time)*1000:.2f} ms")
       return optimal_action
   
   # Detailed profiling
   cProfile.run('profile_mpc_step()', 'mpc_profile.stats')

**Memory Usage Analysis**

.. code-block:: python

   import tracemalloc
   
   # Start memory profiling
   tracemalloc.start()
   
   # Run MPC step
   optimal_action = controller.suggest_action(past_cmas, past_cpps, target_cmas)
   
   # Get memory usage
   current, peak = tracemalloc.get_traced_memory()
   print(f"Current memory: {current / 1024 / 1024:.2f} MB")
   print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")
   
   tracemalloc.stop()

Troubleshooting
===============

**Common Issues**

1. **"No valid control actions found"**
   - Check constraint limits are not too restrictive
   - Verify current operating point is feasible
   - Reduce max_change_per_step limits

2. **"Required variable not found in configuration"**
   - Ensure config contains all required variable names
   - Check spelling of 'spray_rate', 'carousel_speed' etc.
   - Verify soft sensor names match exactly

3. **Slow optimization performance**
   - Reduce discretization_steps (try 3-5)
   - Decrease horizon length
   - Consider upgrading to V2 genetic algorithm

4. **Poor control performance**
   - Tune control_effort_lambda weight
   - Check model prediction quality
   - Verify scaler consistency between training and deployment

5. **Memory issues on GPU**
   - Add torch.cuda.empty_cache() after optimization
   - Reduce batch size in candidate evaluation
   - Move data to CPU when not needed

**Debug Mode**

.. code-block:: python

   # Enable detailed logging
   import logging
   logging.basicConfig(level=logging.DEBUG)
   
   # Add debug prints in MPC loop
   controller.debug_mode = True  # If implemented
   optimal_action = controller.suggest_action(past_cmas, past_cpps, target_cmas)

Limitations
===========

**Algorithmic Limitations**

* **Exponential complexity**: O(n^k) scaling limits real-time use
* **Discrete optimization**: Suboptimal compared to continuous methods
* **No uncertainty handling**: Point predictions without confidence intervals
* **Hard constraints**: Binary feasible/infeasible decisions

**Implementation Limitations**

* **Fixed grid structure**: Cannot adapt discretization during optimization  
* **Sequential evaluation**: No parallelization of candidate assessment
* **Memory inefficient**: Recreates tensors for each candidate
* **Limited error recovery**: Basic fallback mechanisms

**Recommended Upgrades**

Consider V2 implementation for production use:

* Genetic algorithm optimization: O(pg) complexity
* Uncertainty quantification: Monte Carlo dropout predictions
* Soft constraint handling: Penalty function approach
* Integral action: Offset-free control with disturbance estimation

See Also
========

* :doc:`model_architecture` - Neural network prediction model
* :doc:`plant_simulator` - Process simulation for testing
* :doc:`../tutorials/mpc_fundamentals` - Understanding MPC principles
* :doc:`../examples/control_tuning` - Practical tuning examples