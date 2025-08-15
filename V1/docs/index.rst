.. PharmaControl V1 documentation master file

===========================================
PharmaControl V1: Transformer-based MPC
===========================================

.. image:: https://img.shields.io/badge/version-1.0.0-blue.svg
   :target: https://github.com/pharmacontrol/pharmacontrol
   :alt: Version

.. image:: https://img.shields.io/badge/python-3.9+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License

Welcome to PharmaControl V1
============================

PharmaControl V1 is a prototype implementation of transformer-based Model Predictive Control 
for pharmaceutical continuous granulation processes. This educational implementation demonstrates 
the integration of modern deep learning techniques with classical process control for 
pharmaceutical manufacturing applications.

.. note::
   This is the V1 prototype implementation designed for educational and research purposes.
   For production applications, consider the more robust V2 implementation with uncertainty 
   quantification and genetic optimization.

Quick Start
===========

Installation
------------

.. code-block:: bash

   cd V1/
   source .venv/bin/activate
   pip install -e .

Basic Usage
-----------

.. code-block:: python

   from V1.src import plant_simulator, model_architecture, mpc_controller
   import torch
   
   # Create process simulator
   plant = plant_simulator.AdvancedPlantSimulator()
   
   # Load trained model
   model = torch.load('data/best_predictor_model.pth')
   
   # Create MPC controller
   controller = mpc_controller.MPCController(model, config, constraints, scalers)
   
   # Run control step
   cpps = {'spray_rate': 120.0, 'air_flow': 500.0, 'carousel_speed': 30.0}
   state = plant.step(cpps)

Key Features
============

🧠 **Transformer-based Prediction**
   Neural sequence-to-sequence model with attention mechanisms for multi-step process prediction

🎛️ **Model Predictive Control**
   Discrete optimization-based MPC with constraint handling and cost function optimization

🏭 **Realistic Process Simulation**
   High-fidelity granulation simulator with time delays, disturbances, and process interactions

📚 **Educational Focus**
   Clear documentation and progressive learning materials for understanding MPC fundamentals

Process Variables
=================

**Critical Material Attributes (CMAs)**

* **d50**: Median particle size distribution (μm)
* **lod**: Loss on drying/moisture content (%)

**Critical Process Parameters (CPPs)**

* **spray_rate**: Liquid binder spray rate (g/min)
* **air_flow**: Fluidization air flow rate (m³/h)
* **carousel_speed**: Carousel rotation speed (rpm)
* **specific_energy**: Calculated soft sensor (spray_rate × carousel_speed / 1000)
* **froude_number_proxy**: Calculated soft sensor (carousel_speed² / 9.81)

Architecture Overview
=====================

.. image:: _static/v1_architecture.svg
   :alt: V1 Architecture Diagram
   :align: center
   :width: 100%

The V1 architecture consists of four main components:

1. **Plant Simulator** - Realistic granulation process dynamics
2. **Model Architecture** - Transformer encoder-decoder for prediction
3. **MPC Controller** - Discrete optimization with grid search
4. **Dataset** - Time series sequence extraction for training

Educational Notebooks
=====================

The V1 implementation includes a 5-part educational series:

1. **Advanced Process Simulation and Theory** - Process fundamentals and simulation
2. **Data Wrangling and Hybrid Preprocessing** - Data preparation and feature engineering
3. **Predictive Model Training and Validation** - Transformer training and validation
4. **Robust Model Predictive Control System** - MPC implementation and tuning
5. **Closed Loop Simulation and Performance Analysis** - System integration and analysis

API Documentation
=================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules/plant_simulator
   modules/model_architecture
   modules/mpc_controller
   modules/dataset
   tutorials/index
   examples/index
   api/index

Performance Characteristics
===========================

**Computational Complexity**
   O(n^k) where n = discretization steps, k = control variables

**Typical Performance**
   * 3 variables × 5 steps = 125 candidates (✅ Real-time capable)
   * 3 variables × 10 steps = 1,000 candidates (⚠️ Slower)
   * 5 variables × 10 steps = 100,000 candidates (❌ Not real-time)

**Memory Requirements**
   * Model: ~1-5 MB (depending on architecture)
   * Dataset: Scales with sequence length and dataset size
   * GPU memory: ~500 MB typical for batch processing

Limitations and V2 Evolution
=============================

**V1 Limitations (Educational Design)**

* Exhaustive grid search with exponential complexity
* No uncertainty quantification in predictions
* Basic constraint handling with hard filtering
* No integral action for steady-state offset elimination

**V2 Improvements (Production Ready)**

* Genetic algorithm optimization with O(pg) complexity
* Probabilistic predictions with Monte Carlo dropout
* Soft constraint handling with penalty functions
* Integral action and Kalman state estimation

Getting Help
============

**Documentation**
   * :doc:`tutorials/index` - Step-by-step tutorials
   * :doc:`examples/index` - Practical examples
   * :doc:`api/index` - Complete API reference

**Community**
   * GitHub Issues: Report bugs and request features
   * Discussions: Ask questions and share experiences

**Citation**
   If you use PharmaControl V1 in your research, please cite:

   .. code-block:: bibtex

      @software{pharmacontrol_v1,
        title = {PharmaControl V1: Transformer-based MPC for Pharmaceutical Granulation},
        author = {PharmaControl Development Team},
        version = {1.0.0},
        year = {2024},
        url = {https://github.com/pharmacontrol/pharmacontrol}
      }

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. toctree::
   :hidden:
   :maxdepth: 1

   installation
   quickstart
   tutorials/index
   examples/index
   api/index
   changelog
   license