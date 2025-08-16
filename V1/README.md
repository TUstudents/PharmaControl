# PharmaControl V1: Transformer-based MPC

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/pharmacontrol/pharmacontrol)
[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-CC--BY--NC--SA--4.0-green.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

A prototype implementation of transformer-based Model Predictive Control for pharmaceutical continuous granulation processes. This educational implementation demonstrates the integration of modern deep learning techniques with classical process control for pharmaceutical manufacturing applications.

## Overview

PharmaControl V1 is a prototype end-to-end system that:
- **Simulates** a realistic pharmaceutical granulation plant with nonlinear dynamics and disturbances
- **Trains** a Transformer-based predictive model to forecast process behavior  
- **Implements** a discrete optimization-based MPC controller with constraint handling
- **Provides** comprehensive performance analysis and educational materials

> **Note**: This is the V1 prototype implementation designed for educational and research purposes. For production applications, consider the more robust [V2 implementation](../V2/) with uncertainty quantification and genetic optimization.

## Project Structure

```
V1/
├── README.md                          # This file
├── pyproject.toml                     # UV package configuration
├── uv.lock                           # Dependency lock file
├── data/                             # Generated data and trained models
│   ├── granulation_data.csv          # Raw simulation data
│   ├── train_data_raw.csv            # Training set (unscaled)
│   ├── validation_data_raw.csv       # Validation set (unscaled)
│   ├── test_data_raw.csv             # Test set (unscaled)
│   ├── best_predictor_model.pth      # Trained transformer model
│   ├── model_scalers.joblib          # Data scalers for inference
│   ├── optuna_study.pkl              # Hyperparameter optimization results
│   └── training_log.csv              # Training metrics history
├── docs/                             # Sphinx documentation
│   ├── _build/html/index.html        # Built documentation
│   ├── modules/                      # Module documentation
│   ├── api/                          # API reference
│   └── conf.py                       # Sphinx configuration
├── notebooks/                        # Educational Jupyter notebooks
│   ├── 01_Advanced_Process_Simulation_and_Theory.ipynb
│   ├── 02_Data_Wrangling_and_Hybrid_Preprocessing.ipynb
│   ├── 03_Predictive_Model_Training_and_Validation.ipynb
│   ├── 04_Robust_Model_Predictive_Control_System.ipynb
│   └── 05_Closed_Loop_Simulation_and_Performance_Analysis.ipynb
└── src/                              # Core implementation modules
    ├── __init__.py                   # Package initialization
    ├── plant_simulator.py            # Advanced granulation process simulator
    ├── dataset.py                    # PyTorch dataset for time-series
    ├── model_architecture.py         # Transformer-based predictive model
    └── mpc_controller.py             # Model Predictive Controller
```

## Key Features

### 🧠 **Transformer-based Prediction**
Neural sequence-to-sequence model with attention mechanisms for multi-step process prediction

### 🎛️ **Model Predictive Control** 
Discrete optimization-based MPC with constraint handling and cost function optimization

### 🏭 **Realistic Process Simulation**
High-fidelity granulation simulator with time delays, disturbances, and process interactions

### 📚 **Educational Focus**
Clear documentation and progressive learning materials for understanding MPC fundamentals

## Process Variables

**Critical Material Attributes (CMAs)**
- **d50**: Median particle size distribution (μm)
- **lod**: Loss on drying/moisture content (%)

**Critical Process Parameters (CPPs)**  
- **spray_rate**: Liquid binder spray rate (g/min)
- **air_flow**: Fluidization air flow rate (m³/h)
- **carousel_speed**: Carousel rotation speed (rpm)
- **specific_energy**: Calculated soft sensor (spray_rate × carousel_speed / 1000)
- **froude_number_proxy**: Calculated soft sensor (carousel_speed² / 9.81)

## Quick Start

### Installation

**Prerequisites**: Python 3.12+ and [uv](https://docs.astral.sh/uv/) package manager

```bash
# Navigate to V1 directory
cd V1/

# Create virtual environment and install dependencies
uv venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

**Alternative with pip**:
```bash
cd V1/
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Basic Usage

```python
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
```

### Educational Notebooks

Execute the notebooks in sequence:

1. **Advanced Process Simulation and Theory** - Process fundamentals and simulation
2. **Data Wrangling and Hybrid Preprocessing** - Data preparation and feature engineering  
3. **Predictive Model Training and Validation** - Transformer training and validation
4. **Robust Model Predictive Control System** - MPC implementation and tuning
5. **Closed Loop Simulation and Performance Analysis** - System integration and analysis

Each notebook is self-contained with detailed explanations and can be run independently after the data generation step.

## Documentation

- **📖 [Full Documentation](docs/_build/html/index.html)** - Complete API reference and tutorials
- **🔍 [API Reference](docs/api/index.html)** - Detailed module documentation
- **📊 [Architecture Overview](docs/_build/html/index.html#architecture-overview)** - System design and components

## Architecture Overview

The V1 architecture consists of four main components:

1. **Plant Simulator** - Realistic granulation process dynamics
2. **Model Architecture** - Transformer encoder-decoder for prediction  
3. **MPC Controller** - Discrete optimization with grid search
4. **Dataset** - Time series sequence extraction for training

## Performance Characteristics

**Computational Complexity**
- O(n^k) where n = discretization steps, k = control variables

**Typical Performance**
- 3 variables × 5 steps = 125 candidates (✅ Real-time capable)
- 3 variables × 10 steps = 1,000 candidates (⚠️ Slower)  
- 5 variables × 10 steps = 100,000 candidates (❌ Not real-time)

**Memory Requirements**
- Model: ~1-5 MB (depending on architecture)
- Dataset: Scales with sequence length and dataset size
- GPU memory: ~500 MB typical for batch processing

## Limitations and V2 Evolution

**V1 Limitations (Educational Design)**
- Exhaustive grid search with exponential complexity
- No uncertainty quantification in predictions
- Basic constraint handling with hard filtering  
- No integral action for steady-state offset elimination

**V2 Improvements (Production Ready)**
- Genetic algorithm optimization with O(pg) complexity
- Probabilistic predictions with Monte Carlo dropout
- Soft constraint handling with penalty functions
- Integral action and Kalman state estimation

➡️ **For production applications, see [V2 Implementation](../V2/)**

## Getting Help

**Documentation**
- [tutorials/index](docs/tutorials/) - Step-by-step tutorials
- [examples/index](docs/examples/) - Practical examples  
- [API Reference](docs/api/index.html) - Complete API reference

**Community**
- GitHub Issues: Report bugs and request features
- Discussions: Ask questions and share experiences

## Citation

If you use PharmaControl V1 in your research, please cite:

```bibtex
@software{pharmacontrol_v1,
  title = {PharmaControl V1: Transformer-based MPC for Pharmaceutical Granulation},
  author = {PharmaControl Development Team},
  version = {1.0.0},
  year = {2024},
  url = {https://github.com/pharmacontrol/pharmacontrol}
}
```

## License

This work is licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

**You are free to:**
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material

**Under the following terms:**
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made
- **NonCommercial** — You may not use the material for commercial purposes
- **ShareAlike** — If you remix, transform, or build upon the material, you must distribute your contributions under the same license

For commercial use, please contact the authors. Please ensure compliance with applicable regulations when adapting for industrial use.