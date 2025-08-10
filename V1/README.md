# PharmaControl-Pro

A comprehensive, AI-powered Model Predictive Control (MPC) system for pharmaceutical continuous granulation processes. This project demonstrates the integration of advanced machine learning (Transformer models), control theory, and industrial process simulation.

## Overview

PharmaControl-Pro is a complete end-to-end system that:
- Simulates a realistic pharmaceutical granulation plant with nonlinear dynamics and disturbances
- Trains a Transformer-based predictive model to forecast process behavior
- Implements a robust MPC controller that autonomously steers the process to desired targets
- Provides comprehensive performance analysis and visualization tools

## Project Structure

```
PharmaControl-Pro/
├── README.md
├── requirements.txt
├── data/
│   └── granulation_data.csv          # Generated simulation data
├── notebooks/
│   ├── 01_Advanced_Process_Simulation_and_Theory.ipynb
│   ├── 02_Data_Wrangling_and_Hybrid_Preprocessing.ipynb
│   ├── 03_Predictive_Model_Training_and_Validation.ipynb
│   ├── 04_Robust_Model_Predictive_Control_System.ipynb
│   └── 05_Closed_Loop_Simulation_and_Performance_Analysis.ipynb
└── src/
    ├── plant_simulator.py             # Advanced plant simulator
    ├── dataset.py                     # PyTorch dataset for time-series
    ├── model_architecture.py          # Transformer-based predictive model
    ├── mpc_controller.py              # Model Predictive Controller
    └── utils.py                       # Utility functions
```

## Key Features

### 🏭 Realistic Process Simulation
- **Nonlinear Dynamics**: Captures saturation effects and complex interactions
- **Time Lags**: Models material transport delays through the equipment
- **Process Interactions**: d50 particle size affects drying efficiency (LOD)
- **Disturbances**: Simulates filter clogging over time
- **Measurement Noise**: Adds realistic sensor uncertainty

### 🧠 Advanced Machine Learning
- **Transformer Architecture**: Encoder-decoder model with attention mechanisms
- **Hybrid Modeling**: Combines data-driven learning with physics-informed soft sensors
- **Custom Loss Function**: Weighted horizon MSE that prioritizes long-term accuracy
- **Hyperparameter Optimization**: Systematic tuning using Optuna
- **Time-Series Best Practices**: Chronological splitting to prevent data leakage

### 🎛️ Robust Control System
- **Model Predictive Control**: Receding horizon optimization with constraints
- **Safety Constraints**: Enforces equipment limits and operational bounds
- **Multi-Objective Optimization**: Balances target tracking and control effort
- **Real-Time Decision Making**: Evaluates multiple candidate actions per control cycle

### 📊 Comprehensive Analysis
- **Quantitative Metrics**: Settling time, overshoot, and steady-state error
- **Interactive Visualizations**: Real-time plotting of process variables and control actions
- **Performance Benchmarking**: Comparison against simpler baseline models

## Process Variables

### Critical Process Parameters (CPPs) - Control Inputs
- `spray_rate` (g/min): Liquid spray rate affecting granule size
- `air_flow` (m³/h): Drying air flow rate affecting moisture content
- `carousel_speed` (rph): Carousel rotation speed affecting residence time

### Critical Material Attributes (CMAs) - Process Outputs
- `d50` (μm): Median particle size of granules
- `LOD` (%): Loss on Drying (residual moisture content)

### Soft Sensors - Physics-Informed Features
- `specific_energy`: Energy input proxy based on spray rate and speed
- `froude_number_proxy`: Dimensionless mixing intensity indicator

## Getting Started

### Prerequisites
```bash
# Python 3.8+
pip install torch pandas numpy matplotlib scikit-learn optuna tqdm joblib
```

### Installation
```bash
git clone <repository-url>
cd PharmaControl-Pro
pip install -r requirements.txt
```

### Usage
Execute the notebooks in sequence:

1. **Notebook 1**: Build and test the advanced plant simulator
2. **Notebook 2**: Generate data and perform preprocessing with soft sensors
3. **Notebook 3**: Train and validate the Transformer predictive model
4. **Notebook 4**: Implement and test the MPC controller
5. **Notebook 5**: Run closed-loop simulation and performance analysis

Each notebook is self-contained with detailed explanations and can be run independently after the data generation step.

## Technical Highlights

### Model Architecture
- **Encoder**: Processes historical CMAs and CPPs using self-attention
- **Decoder**: Generates future predictions using cross-attention with planned control actions
- **Positional Encoding**: Injects temporal information into sequences
- **Causal Masking**: Prevents information leakage during training

### MPC Algorithm
1. **Generate Candidates**: Create lattice of possible future control sequences
2. **Apply Constraints**: Filter out unsafe or invalid actions
3. **Predict Outcomes**: Use trained model to forecast results for each candidate
4. **Optimize**: Select action minimizing weighted cost function
5. **Execute**: Apply first step of optimal plan (receding horizon)

### Data Processing
- **Chronological Splitting**: 70% train / 15% validation / 15% test
- **Feature Scaling**: MinMaxScaler fitted only on training data
- **Sequence Generation**: Sliding window approach for time-series samples
- **Batch Processing**: Efficient PyTorch DataLoader implementation

## Performance Metrics

The system is evaluated using industrial control standards:
- **Settling Time**: Time to reach and stay within ±5% of target
- **Overshoot**: Maximum deviation beyond target as percentage
- **Steady-State Error**: Final average error after stabilization

## Applications

This framework can be adapted for various process control applications:
- Pharmaceutical manufacturing (tablets, capsules, granulation)
- Chemical process control (reactors, separations, crystallization)
- Food and beverage production (mixing, fermentation, drying)
- Advanced materials processing (polymers, ceramics, composites)

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is provided for educational and research purposes. Please ensure compliance with applicable regulations when adapting for industrial use.

## Acknowledgments

This project demonstrates advanced concepts in:
- Process Control Theory and Model Predictive Control
- Deep Learning for Time-Series Forecasting
- Hybrid Modeling and Physics-Informed Machine Learning
- Industrial Process Simulation and Digital Twins
- Pharmaceutical Manufacturing and Quality by Design (QbD)

## Citation

If you use this work in your research, please cite:
```
PharmaControl-Pro: An AI-Powered Model Predictive Control System 
for Pharmaceutical Continuous Granulation Processes
```