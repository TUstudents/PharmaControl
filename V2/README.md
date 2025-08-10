# RobustMPC-Pharma V2: Industrial-Grade Control System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Educational](https://img.shields.io/badge/license-Educational-green.svg)](LICENSE)
[![Version: 2.0.0](https://img.shields.io/badge/version-2.0.0-brightgreen.svg)](robust_mpc/__init__.py)

**Next-Generation Model Predictive Control with Uncertainty Quantification and Adaptive Intelligence**

## 🚀 What's New in V2

V2 represents a complete architectural evolution from prototype to production-ready industrial control system:

### **From Reactive to Proactive Control**
- **V1**: Reactive controller responding to current state
- **V2**: Proactive system that anticipates uncertainty and learns from experience

### **Core Architectural Improvements**
- **🔧 Modular Design**: Composable `robust_mpc` library with dependency injection
- **📊 Uncertainty Awareness**: Every prediction includes confidence bounds  
- **🔍 State Estimation**: Kalman filtering eliminates sensor noise issues
- **🧬 Intelligent Optimization**: Genetic algorithms replace brute-force search
- **⚖️ Offset-Free Control**: Integral action eliminates steady-state errors
- **🛡️ Industrial Robustness**: Formal guarantees and safety constraints

## 📁 Project Structure

```
V2/
├── 📖 DESIGN_DOCUMENT.md          # Comprehensive technical specification
├── 📋 README.md                   # This file
├── 📦 requirements.txt            # Enhanced dependencies
├── 🏗️ robust_mpc/                 # Core library (production-ready)
│   ├── __init__.py               # Library interface & metadata
│   ├── estimators.py             # Kalman filtering & state estimation  
│   ├── models.py                 # Probabilistic prediction models
│   ├── optimizers.py             # Advanced optimization algorithms
│   └── core.py                   # Main controller orchestration
├── 📓 notebooks/                  # Progressive tutorial series
│   ├── V2-1_State_Estimation_for_Stable_Control.ipynb
│   ├── V2-2_Probabilistic_Modeling_for_Uncertainty_Awareness.ipynb
│   ├── V2-3_Advanced_Optimization_with_Genetic_Algorithms.ipynb
│   ├── V2-4_The_Robust_MPC_Core_Integrating_Intelligence.ipynb
│   └── V2-5_V2_vs_V1_Showdown_Stress_Test_Comparison.ipynb
├── 💾 data/                       # Generated datasets & artifacts
├── 🎯 models/                     # Trained model checkpoints
└── 🧪 tests/                      # Comprehensive test suite
```

## 🎯 Key Components

### **State Estimation (`estimators.py`)**
```python
from robust_mpc import KalmanStateEstimator

estimator = KalmanStateEstimator(
    transition_matrix=A_matrix,
    control_matrix=B_matrix, 
    initial_state_mean=initial_state,
    process_noise_std=1.0,
    measurement_noise_std=15.0
)

# Clean, filtered state from noisy measurements
filtered_state = estimator.estimate(noisy_measurement, control_input)
uncertainty = estimator.get_uncertainty()
```

### **Probabilistic Modeling (`models.py`)**
```python
from robust_mpc import ProbabilisticTransformer

model = ProbabilisticTransformer(
    cma_features=2, 
    cpp_features=5,
    mc_samples=50
)

# Predictions with uncertainty quantification
mean, std = model.predict_distribution(past_data, future_actions)
risk_adjusted = mean + beta * std  # Upper confidence bound
```

### **Advanced Optimization (`optimizers.py`)**
```python
from robust_mpc import GeneticOptimizer

optimizer = GeneticOptimizer(
    population_size=50,
    generations=20,
    mutation_rate=0.1
)

# Intelligent search through complex action spaces
best_action = optimizer.optimize(fitness_function, bounds, constraints)
pareto_front = optimizer.optimize_pareto(multi_objective_fitness, bounds)
```

### **Robust Controller (`core.py`)**
```python
from robust_mpc import RobustMPCController

controller = RobustMPCController(
    model=probabilistic_model,
    estimator=kalman_filter, 
    optimizer_class=GeneticOptimizer,
    config=mpc_config,
    scalers=data_scalers
)

# Main control loop with full robustness stack
optimal_action = controller.suggest_action(noisy_measurement, control_input, setpoint)

# Get comprehensive performance metrics
metrics = controller.get_performance_metrics()
```

## 📚 Progressive Learning Series

### **Notebook V2-1: State Estimation Foundations**
**Status:** ✅ **Complete**
- Problem: Why controlling noise leads to instability
- Solution: Kalman Filter theory and implementation  
- Deliverable: `KalmanStateEstimator` class

### **Notebook V2-2: Probabilistic Modeling** 
**Status:** ✅ **Complete**
- Uncertainty quantification with Monte Carlo Dropout
- Epistemic vs. aleatoric uncertainty  
- Risk-aware prediction models
- Deliverable: `ProbabilisticTransformer` class

### **Notebook V2-3: Advanced Optimization**
**Status:** ✅ **Complete**
- Genetic algorithms for control optimization
- Complex action space exploration
- Constraint handling in evolutionary algorithms
- Deliverable: `GeneticOptimizer` class

### **Notebook V2-4: Robust MPC Integration**  
**Status:** ✅ **Complete**
- Complete controller assembly with all components
- Integral action for offset-free control
- Risk-adjusted cost functions with uncertainty quantification
- Deliverable: `RobustMPCController` class

### **Notebook V2-5: V1 vs V2 Showdown**
**Status:** ✅ **Complete**  
- Head-to-head V2 vs V1 performance comparison
- Comprehensive stress testing under disturbances
- Quantitative robustness analysis with metrics
- Deliverable: Complete validation and performance proof

## ⚡ Quick Start

### **Installation**
```bash
git clone <repository-url>
cd PharmaControl/V2
pip install -r requirements.txt
```

### **Library Usage**
```python
import robust_mpc

# Print library information
robust_mpc.print_library_info()

# Get default configuration
config = robust_mpc.get_default_config()

# Import and use components as they become available
from robust_mpc import KalmanStateEstimator     # ✅ Available (V2-1)
from robust_mpc import ProbabilisticTransformer # ✅ Available (V2-2)
from robust_mpc import GeneticOptimizer         # ✅ Available (V2-3)
from robust_mpc import RobustMPCController      # ✅ Available (V2-4)
```

### **Run the Controller Application**
```bash
# Basic usage
python run_controller.py

# With custom configuration
python run_controller.py --config my_config.yaml

# Fast simulation (no delays)
python run_controller.py --no-realtime --steps 500

# Help and options
python run_controller.py --help
```

### **Run Tests**
```bash
cd tests
python test_library_structure.py
# Or with pytest
pytest test_library_structure.py -v
```

## 🐳 Docker Deployment

### **Build Container**
```bash
docker build -t robust-mpc-pharma:v2 .
```

### **Run Container**
```bash
# Interactive mode
docker run -it --rm robust-mpc-pharma:v2

# Background mode with custom config
docker run -d --name pharma-control \
  -v $(pwd)/my_config.yaml:/app/config.yaml \
  robust-mpc-pharma:v2

# With volume mounts for data persistence
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  robust-mpc-pharma:v2
```

### **Development with Docker Compose** (Optional)
Create `docker-compose.yml`:
```yaml
version: '3.8'
services:
  robust-mpc:
    build: .
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./config.yaml:/app/config.yaml
    environment:
      - PYTHONPATH=/app:/app/robust_mpc
    command: python run_controller.py --config config.yaml
```

Run with: `docker-compose up`

## 📋 Configuration Management

### **Configuration File Structure**
The `config.yaml` file provides centralized control over all system parameters:

```yaml
# Process variables and constraints
process:
  cma_names: ['d50', 'lod']
  cpp_names: ['spray_rate', 'air_flow', 'carousel_speed']

# MPC controller tuning
mpc:
  horizon: 50                # Planning horizon
  integral_gain: 0.05        # Offset-free control strength
  risk_beta: 1.5            # Risk aversion (0=neutral, >0=conservative)
  
# Genetic algorithm optimization
  population_size: 40
  generations: 15

# Simulation settings
simulation:
  total_steps: 1000
  step_interval_seconds: 1.0
  target_setpoint: {d50: 380.0, lod: 1.8}
```

### **Environment-Specific Configurations**
```bash
# Development (fast simulation)
python run_controller.py --config config_dev.yaml

# Production (real-time operation)  
python run_controller.py --config config_prod.yaml

# Testing (minimal steps)
python run_controller.py --config config_test.yaml
```

## 🚀 Production Deployment

### **Prerequisites**
- Python 3.9+ environment
- Required dependencies (see `requirements.txt`)
- Pre-trained model files (optional for demo)
- Configuration file tuned for your process

### **Installation Methods**

#### **Method 1: Direct Installation**
```bash
# Clone repository
git clone <repository-url>
cd PharmaControl/V2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Run application
python run_controller.py
```

#### **Method 2: Package Installation**
```bash
# Install as package
pip install -e .

# Run from anywhere
robust-mpc --config /path/to/config.yaml
# or
pharma-control --config /path/to/config.yaml
```

#### **Method 3: Container Deployment**
```bash
# Production container with persistent data
docker run -d --name pharma-mpc \
  --restart unless-stopped \
  -v /host/data:/app/data \
  -v /host/models:/app/models \
  -v /host/config.yaml:/app/config.yaml \
  -e PYTHONPATH=/app:/app/robust_mpc \
  robust-mpc-pharma:v2
```

### **Monitoring and Logging**
```bash
# View container logs
docker logs -f pharma-mpc

# Monitor performance metrics
docker exec pharma-mpc python -c "
from robust_mpc.core import RobustMPCController
# Get performance metrics programmatically
"

# Health check
docker exec pharma-mpc python -c "import robust_mpc; print('System OK')"
```

## 🔧 Development Setup

### **Development Environment**
```bash
# Clone with full development setup
git clone <repository-url>
cd PharmaControl/V2

# Install with development dependencies
pip install -e ".[dev,notebooks]"

# Set up pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v --cov=robust_mpc

# Code formatting
black robust_mpc/
isort robust_mpc/

# Type checking
mypy robust_mpc/
```

### **Jupyter Development**
```bash
# Start Jupyter Lab
jupyter lab

# Or with Docker for isolated environment
docker run -p 8888:8888 -v $(pwd):/work \
  robust-mpc-pharma:v2 \
  jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

## 🎯 Performance Targets vs V1

| Metric | V1 Baseline | V2 Target | Improvement |
|--------|-------------|-----------|-------------|
| **Settling Time** | - | -30% | Advanced optimization |
| **Steady-State Error** | - | -90% | Integral action |  
| **Control Variance** | - | -50% | State filtering |
| **Disturbance Robustness** | - | +80% | Uncertainty awareness |
| **Computational Speed** | - | +60% | Intelligent search |

## 🛠️ Advanced Features (Future Versions)

### **V2.1: Adaptive Intelligence**
- Online model recalibration
- Performance degradation detection  
- Self-tuning parameters

### **V2.2: Economic Optimization**
- Multi-objective optimization (quality vs cost vs throughput)
- Supply chain integration
- Dynamic economic models

### **V2.3: Formal Guarantees** 
- Tube MPC for constraint satisfaction
- Robust invariant sets
- Safety-critical compliance

## 🛠️ Troubleshooting

### **Common Issues and Solutions**

#### **Import Errors**
```bash
# Error: ModuleNotFoundError: No module named 'robust_mpc'
# Solution: Ensure correct Python path
export PYTHONPATH=$PYTHONPATH:/path/to/PharmaControl/V2

# Or install as package
pip install -e .
```

#### **Missing Dependencies**
```bash
# Error: No module named 'pykalman' or 'deap'
# Solution: Install all requirements
pip install -r requirements.txt

# For development dependencies
pip install -e ".[dev,notebooks]"
```

#### **Configuration Issues**
```bash
# Error: Configuration file not found
# Solution: Specify correct path
python run_controller.py --config /full/path/to/config.yaml

# Or create default config
python -c "
import yaml
from run_controller import get_default_config
with open('config.yaml', 'w') as f:
    yaml.dump(get_default_config(), f, default_flow_style=False)
"
```

#### **Performance Issues**
```bash
# Issue: Slow optimization
# Solution: Reduce GA parameters in config.yaml
mpc:
  population_size: 20    # Reduce from 40
  generations: 10        # Reduce from 15

# Issue: Memory usage
# Solution: Reduce model complexity
model:
  hyperparameters:
    d_model: 32          # Reduce from 64
    mc_samples: 10       # Reduce from 25
```

#### **Docker Issues**
```bash
# Issue: Container fails to start
# Solution: Check logs and rebuild
docker logs <container-name>
docker build --no-cache -t robust-mpc-pharma:v2 .

# Issue: Permission errors
# Solution: Fix file permissions
chmod -R 755 /path/to/PharmaControl/V2
```

### **Debugging Mode**
```bash
# Enable verbose logging
python run_controller.py --config config.yaml --debug

# Run with Python debugger
python -m pdb run_controller.py

# Profile performance
python -m cProfile -o profile.stats run_controller.py
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('time').print_stats(10)"
```

## 🧪 Testing and Validation

### **Unit Tests**
```bash
# Run basic library tests
python tests/test_library_structure.py

# Run with pytest for detailed output
pytest tests/ -v --cov=robust_mpc --cov-report=html

# Test specific components
pytest tests/test_core.py::test_robust_mpc_controller -v
```

### **Integration Tests**
```bash
# Test full application
python run_controller.py --no-realtime --steps 100

# Test with different configurations
python run_controller.py --config config_test.yaml --steps 50

# Container integration test
docker run --rm robust-mpc-pharma:v2 python run_controller.py --steps 10
```

### **Performance Validation**
```bash
# Run V2-5 comparison notebook
jupyter nbconvert --execute notebooks/V2-5_V2_vs_V1_Showdown_Stress_Test_Comparison.ipynb

# Generate performance report
python -c "
from robust_mpc.core import RobustMPCController
# Run automated performance assessment
"
```

## 🔬 Technical Innovations

### **Risk-Aware Optimization**
```python
# Traditional MPC: minimize cost(prediction_mean)
# V2 MPC: minimize cost(prediction_mean + β * prediction_std)
risk_adjusted_cost = tracking_error + β * prediction_uncertainty + λ * control_effort
```

### **Integral Action for Industrial Use**
```python
# Eliminates steady-state offset from unmeasured disturbances
tracking_error = setpoint - filtered_measurement
disturbance_estimate += integral_gain * tracking_error
corrected_prediction = model_prediction + disturbance_estimate
```

### **Intelligent Action Parameterization**
```python
# V1: Constant actions [spray_rate, air_flow, speed]
# V2: Complex sequences [initial, ramp_rate, hold_time, final, ...]
chromosome = encode_ramp_sequence(initial_vals, ramp_rates, durations)
```

## 📊 Educational Impact

This V2 series provides **graduate-level** understanding of:
- ✅ State-space estimation and Kalman filtering
- ✅ Uncertainty quantification in control systems  
- ✅ Evolutionary optimization for control
- ✅ Industrial MPC design patterns and integration
- ✅ Robustness analysis and validation methodology

## 🤝 Contributing

This is an educational project demonstrating advanced control concepts. Contributions welcome for:
- 📝 Documentation improvements
- 🧪 Additional test cases
- 🔧 Algorithm implementations  
- 📊 Visualization enhancements

## 📄 License

Educational/Research use. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

Built upon fundamental concepts from:
- **Control Theory**: Kalman filtering, MPC, robust control
- **Machine Learning**: Uncertainty quantification, Bayesian methods
- **Optimization**: Evolutionary algorithms, multi-objective optimization  
- **Industrial Practice**: Pharmaceutical manufacturing, Quality by Design

---

**🎯 Mission**: Transform pharmaceutical manufacturing through intelligent, adaptive, uncertainty-aware process control.

**🔬 Vision**: Bridge the gap between academic control theory and industrial reality with production-ready, safety-critical systems.