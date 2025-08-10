# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PharmaControl is an advanced process control system evolution for pharmaceutical continuous granulation processes, demonstrating the complete journey from research prototype to industrial-grade implementation with autonomous intelligence. The project contains three major versions:

- **V1**: Prototype system with Transformer-based prediction and basic MPC
- **V2**: Industrial-grade system with uncertainty quantification, Kalman filtering, and genetic optimization
- **V3**: Autonomous system with online learning, explainable AI, and reinforcement learning (in development)

## Development Commands

### Package Management
All versions use uv as package manager with `.venv` virtual environments:

```bash
# Setup virtual environment and install dependencies for any version
cd V1/  # or V2/ or V3/
uv venv .venv
source .venv/bin/activate
uv pip install -e .

# Install with development dependencies (V2)
cd V2/
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev,notebooks]"

# Sync dependencies from lock file (with activated environment)
source .venv/bin/activate
uv sync
```

### Testing
```bash
# V2 comprehensive testing
cd V2/
source .venv/bin/activate
pytest tests/ -v --cov=robust_mpc
python tests/test_library_structure.py

# V2 specific component tests
python test_v2_3_completion.py
python test_v2_4_completion.py
```

### Running Controllers
```bash
# V2 production controller
cd V2/
source .venv/bin/activate
python run_controller.py
python run_controller.py --config config.yaml --no-realtime --steps 500

# V1 notebook-based execution
cd V1/
source .venv/bin/activate
jupyter lab  # Run notebooks 01-05 in sequence
```

### Code Quality (V2)
```bash
cd V2/
source .venv/bin/activate
black robust_mpc/
isort robust_mpc/
mypy robust_mpc/
```

## Architecture Overview

### V1 Architecture (Prototype)
- **Monolithic design** with individual classes
- **Core files**: `src/plant_simulator.py`, `src/model_architecture.py`, `src/mpc_controller.py`
- **Notebooks**: 5-part educational series covering simulation, preprocessing, training, MPC, and analysis
- **Data flow**: CSV generation → Transformer training → MPC control loop

### V2 Architecture (Industrial)
- **Modular library design** with `robust_mpc/` package
- **Key modules**:
  - `estimators.py`: KalmanStateEstimator for sensor noise filtering
  - `models.py`: ProbabilisticTransformer with uncertainty quantification
  - `optimizers.py`: GeneticOptimizer for complex action spaces
  - `core.py`: RobustMPCController orchestrating all components
- **Progressive learning**: 5 notebooks building complexity incrementally
- **Docker deployment** ready with production configuration

### V3 Architecture (Autonomous)
- **Service-oriented architecture** with microservices
- **Key services**:
  - `control_agent/`: Main control orchestration
  - `learning_service/`: Online model adaptation
  - `monitoring_xai_service/`: Explainable AI monitoring
- **Core packages**: `autopharm_core/` with control, learning, RL, and XAI modules

## Key Technical Concepts

### Process Variables
- **CPPs (Critical Process Parameters)**: `spray_rate`, `air_flow`, `carousel_speed` (control inputs)
- **CMAs (Critical Material Attributes)**: `d50` (particle size), `LOD` (moisture) (outputs)
- **Soft sensors**: Physics-informed features like `specific_energy`, `froude_number_proxy`

### Control Architecture Evolution
- **V1**: Reactive control with point predictions
- **V2**: Proactive control with uncertainty quantification and integral action
- **V3**: Autonomous control with online learning and explainable decisions

### Data Processing Patterns
- **Chronological splitting**: 70% train / 15% validation / 15% test (prevents temporal leakage)
- **Feature scaling**: MinMaxScaler fitted only on training data
- **Sequence generation**: Sliding window for time-series samples

## Working with the Codebase

### Version Selection Guidelines
- Use **V1** for understanding fundamentals and educational purposes
- Use **V2** for production deployments and industrial applications
- Use **V3** for research into autonomous manufacturing systems

### Common Development Workflows

When working with V2 (most common):
1. Navigate to `V2/` directory
2. Setup environment: `uv venv .venv && source .venv/bin/activate && uv pip install -e ".[dev,notebooks]"`
3. Test with `pytest tests/ -v` (environment activated)
4. Run controller with `python run_controller.py` (environment activated)
5. Modify `robust_mpc/` modules as needed
6. Test changes with existing test suite

When adding new features:
1. Follow the modular design pattern from V2
2. Add comprehensive tests in `tests/`
3. Update configuration in `config.yaml` if needed
4. Validate with performance benchmarks

### Important File Locations
- **V1 data**: `V1/data/` (pre-processed datasets and trained models)
- **V2 library**: `V2/robust_mpc/` (production-ready modules)
- **V3 services**: `V3/services/` (microservice implementations)
- **Configuration**: `V2/config.yaml`, `V2/Dockerfile`
- **Documentation**: Version-specific README files and DESIGN_DOCUMENT.md files

## Development Philosophy

- **Safety first**: All control systems include constraint handling and safety limits
- **Uncertainty awareness**: V2+ explicitly models and uses prediction uncertainty
- **Industrial readiness**: Code follows production patterns with proper error handling
- **Educational value**: Clear documentation and progressive learning materials
- **Reproducible research**: Fixed random seeds and deterministic behavior where possible