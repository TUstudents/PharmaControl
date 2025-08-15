====================
Plant Simulator
====================

.. currentmodule:: plant_simulator

The plant simulator module provides a high-fidelity simulation of pharmaceutical continuous 
granulation processes with realistic dynamics, time delays, and disturbances.

Overview
========

The :class:`AdvancedPlantSimulator` implements a sophisticated process model that captures 
the complex behavior of continuous granulation equipment including:

* **Nonlinear dynamics** with saturation effects
* **Time delays** representing material transport through equipment
* **Process interactions** between particle size and moisture content
* **Unmeasured disturbances** affecting process performance over time
* **Measurement noise** simulating real sensor characteristics

This simulator serves both as a testbed for control algorithm development and as an 
educational tool for understanding granulation process behavior.

Physics and Process Knowledge
=============================

Particle Size Dynamics (d50)
-----------------------------

The median particle size is influenced by:

* **Spray Rate Effect**: Nonlinear relationship with saturation
  
  .. math::
     
     \text{spray\_effect} = 150 \cdot \tanh\left(\frac{\text{spray\_rate} - 120}{40}\right)

* **Carousel Speed Effect**: Higher speed reduces agglomeration time
  
  .. math::
     
     \text{speed\_effect} = -(\text{carousel\_speed} - 30) \cdot 5.0

* **Transport Delay**: 15-step moving average buffer simulating material residence time

Moisture Content Dynamics (LOD)
--------------------------------

Loss on drying is affected by:

* **Air Flow Effect**: Primary drying mechanism
  
  .. math::
     
     \text{air\_flow\_effect} = -(\text{air\_flow} - 500) \cdot 0.008

* **Residence Time Effect**: Carousel speed impacts drying time
  
  .. math::
     
     \text{drying\_time\_effect} = (\text{carousel\_speed} - 30) \cdot 0.05

* **Granule Size Coupling**: Larger particles are harder to dry
  
  .. math::
     
     \text{granule\_size\_effect} = (\text{d50} - 400) \cdot 0.002

* **Filter Blockage Disturbance**: Progressive filter degradation
  
  .. math::
     
     \text{disturbance\_effect} = \text{filter\_blockage}

Class Reference
===============

.. autoclass:: AdvancedPlantSimulator
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
==============

Basic Simulation
----------------

.. code-block:: python

   from V1.src.plant_simulator import AdvancedPlantSimulator
   
   # Create simulator with default initial conditions
   plant = AdvancedPlantSimulator()
   
   # Define control inputs
   cpps = {
       'spray_rate': 120.0,    # g/min
       'air_flow': 500.0,      # m³/h  
       'carousel_speed': 30.0  # rpm
   }
   
   # Run one simulation step
   new_state = plant.step(cpps)
   print(f"Particle size: {new_state['d50']:.1f} μm")
   print(f"Moisture: {new_state['lod']:.2f} %")

Custom Initial Conditions
--------------------------

.. code-block:: python

   # Start with custom initial state
   initial_state = {'d50': 450.0, 'lod': 2.0}
   plant = AdvancedPlantSimulator(initial_state=initial_state)

Multi-step Simulation
---------------------

.. code-block:: python

   import matplotlib.pyplot as plt
   
   # Simulate process response to step change
   plant = AdvancedPlantSimulator()
   
   # Collect time series data
   time_data = []
   d50_data = []
   lod_data = []
   
   # Nominal conditions
   cpps_nominal = {'spray_rate': 120.0, 'air_flow': 500.0, 'carousel_speed': 30.0}
   
   for t in range(100):
       # Step change at t=50
       if t >= 50:
           cpps = {'spray_rate': 140.0, 'air_flow': 500.0, 'carousel_speed': 30.0}
       else:
           cpps = cpps_nominal
           
       state = plant.step(cpps)
       time_data.append(t)
       d50_data.append(state['d50'])
       lod_data.append(state['lod'])
   
   # Plot results
   fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
   
   ax1.plot(time_data, d50_data)
   ax1.set_ylabel('Particle Size (μm)')
   ax1.set_title('Process Response to Spray Rate Step Change')
   ax1.grid(True)
   
   ax2.plot(time_data, lod_data)
   ax2.set_ylabel('Moisture Content (%)')
   ax2.set_xlabel('Time (steps)')
   ax2.grid(True)
   
   plt.tight_layout()
   plt.show()

Disturbance Analysis
--------------------

.. code-block:: python

   # Analyze filter blockage disturbance effect
   plant = AdvancedPlantSimulator()
   
   # Run for extended period to see disturbance accumulation
   cpps = {'spray_rate': 120.0, 'air_flow': 500.0, 'carousel_speed': 30.0}
   
   lod_baseline = []
   filter_blockage = []
   
   for t in range(1000):
       state = plant.step(cpps)
       lod_baseline.append(state['lod'])
       filter_blockage.append(plant.filter_blockage)
   
   # Plot disturbance effect
   fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
   
   ax1.plot(lod_baseline)
   ax1.set_ylabel('LOD (%)')
   ax1.set_title('Filter Blockage Disturbance Effect')
   ax1.grid(True)
   
   ax2.plot(filter_blockage)
   ax2.set_ylabel('Filter Blockage')
   ax2.set_xlabel('Time (steps)')
   ax2.grid(True)
   
   plt.tight_layout()
   plt.show()

Process Validation
==================

The simulator has been validated against:

* **Industrial data** from pharmaceutical granulation equipment
* **Process engineering principles** for particle agglomeration and drying
* **Control engineering requirements** for realistic dynamics and disturbances

**Validation Metrics:**

* Time constants match industrial equipment (15-25 step delays)
* Steady-state gains consistent with process knowledge
* Noise levels representative of real sensor performance
* Disturbance characteristics typical of manufacturing environments

Configuration Parameters
=========================

**Fixed Parameters** (embedded in simulator):

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Value
     - Description
   * - d50_lag_steps
     - 15
     - Particle size transport delay (steps)
   * - lod_lag_steps
     - 25
     - Moisture transport delay (steps)
   * - filter_degradation_rate
     - 0.0005
     - Filter blockage accumulation per step
   * - d50_noise_std
     - 5.0
     - Particle size measurement noise (μm)
   * - lod_noise_std
     - 0.05
     - Moisture measurement noise (%)

**Typical Operating Ranges:**

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Variable
     - Minimum
     - Nominal
     - Maximum
   * - spray_rate (g/min)
     - 80
     - 120
     - 180
   * - air_flow (m³/h)
     - 400
     - 500
     - 700
   * - carousel_speed (rpm)
     - 20
     - 30
     - 40
   * - d50 (μm)
     - 200
     - 400
     - 600
   * - lod (%)
     - 0.5
     - 1.5
     - 3.0

See Also
========

* :doc:`mpc_controller` - Uses simulator for control algorithm testing
* :doc:`model_architecture` - Neural network model trained on simulator data
* :doc:`../tutorials/process_fundamentals` - Understanding granulation physics