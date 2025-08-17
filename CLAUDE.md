# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PharmaControl is an advanced process control system evolution for pharmaceutical continuous granulation processes, demonstrating the complete journey from research prototype to industrial-grade implementation with autonomous intelligence. The project contains three major versions:

- **V1**: Prototype system with Transformer-based prediction and basic MPC
- **V2**: Industrial-grade system with uncertainty quantification, Kalman filtering, and genetic optimization
- **V3**: Autonomous system with online learning, explainable AI, and reinforcement learning (in development)

## Development Commands

### Central Environment Setup
The project now uses a unified central environment that supports all three versions:

```bash
# Activate the central environment (from project root)
source .venv/bin/activate

# Install project with all dependencies and development tools
uv pip install -e ".[dev,notebooks]"

# Sync dependencies from lock file (if available)
uv sync
```

**Key Benefits of Central Environment:**
- ✅ All modules discoverable: `V1.src`, `V2.robust_mpc`, `V3.src.autopharm_core`
- ✅ Cross-module imports working: `from V1.src import plant_simulator`
- ✅ Unified dependency management across all versions
- ✅ Single Python 3.12.3 environment for consistent behavior

### Package Management
Project uses uv as package manager with a central `.venv` virtual environment:

```bash
# Central configuration in pyproject.toml includes:
# - V1.src, V2.robust_mpc, V3.src.autopharm_core packages
# - All dependencies: torch, numpy, scipy, pandas, etc.
# - Development tools: pytest, black, mypy, isort
# - Fixed license format: CC-BY-NC-SA-4.0
# - Python 3.12 consistency across all configurations
```

### Testing
```bash
# Activate central environment first
source .venv/bin/activate

# V2 comprehensive testing
pytest V2/tests/ -v --cov=robust_mpc
python V2/tests/test_library_structure.py

# V2 specific component tests  
python V2/test_v2_3_completion.py
python V2/test_v2_4_completion.py

# Cross-version testing (all tests from central environment)
pytest V1/tests/ V2/tests/ V3/tests/ -v
```

### Running Controllers
```bash
# Activate central environment first
source .venv/bin/activate

# V2 production controller (run from project root)
python V2/run_controller.py
python V2/run_controller.py --config V2/config.yaml --no-realtime --steps 500

# V1 notebook-based execution
jupyter lab  # Navigate to V1/notebooks/ and run 01-05 in sequence

# V3 services (when available)
python V3/services/control_agent/main.py
```

### Code Quality
```bash
# Activate central environment first
source .venv/bin/activate

# Format code across all versions
black V1/ V2/ V3/
isort V1/ V2/ V3/

# Type checking across all versions
mypy V1/ V2/ V3/

# Version-specific formatting
black V2/robust_mpc/  # V2 library only
mypy V2/robust_mpc/   # V2 library only
```

## Architecture Overview

### V1 Architecture (Prototype - Recently Enhanced)
- **Monolithic design** with individual classes and professional documentation
- **Core files**: 
  - `src/plant_simulator.py`: High-fidelity granulation simulator with realistic dynamics
  - `src/model_architecture.py`: Transformer encoder-decoder for sequence prediction
  - `src/mpc_controller.py`: Discrete optimization MPC with grid search (bugs fixed)
  - `src/dataset.py`: Time series dataset with sliding window extraction
  - `src/__init__.py`: Professional package structure with dependency management
- **Notebooks**: 5-part educational series covering simulation, preprocessing, training, MPC, and analysis
- **Data flow**: CSV generation → Transformer training → MPC control loop
- **Recent improvements**: Bug fixes, error handling, comprehensive docstrings

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

**Central Environment Setup (Required First Step):**
1. Navigate to project root: `/home/feynman/projects/PharmaControl`
2. Activate central environment: `source .venv/bin/activate`
3. Verify installation: `python -c "import V1.src, V2.robust_mpc, V3.src.autopharm_core; print('All modules available')"`

When working with V1 (educational/prototype):
1. From project root with activated environment
2. Use Jupyter notebooks: `jupyter lab` → Navigate to `V1/notebooks/`
3. Execute notebooks in sequence: 01 → 02 → 03 → 04 → 05
4. Modify `V1/src/` modules following professional documentation standards
5. Test changes using notebook examples and validation cells
6. Import pattern: `from V1.src import plant_simulator, model_architecture`

When working with V2 (production):
1. From project root with activated environment
2. Test with `pytest V2/tests/ -v --cov=robust_mpc`
3. Run controller with `python V2/run_controller.py --config V2/config.yaml`
4. Modify `V2/robust_mpc/` modules as needed
5. Test changes with existing test suite
6. Import pattern: `from V2.robust_mpc import core, models, estimators`

When working with V3 (autonomous):
1. From project root with activated environment
2. Run services: `python V3/services/control_agent/main.py`
3. Modify `V3/src/autopharm_core/` modules as needed
4. Import pattern: `from V3.src.autopharm_core import control, learning, rl`

When adding new features:
1. Follow the modular design pattern from V2
2. Add comprehensive tests in version-specific `tests/` directories
3. Update configuration files as needed (`V2/config.yaml`, etc.)
4. Validate with performance benchmarks using central environment
5. Follow established docstring standards for all new code
6. Ensure cross-version compatibility when importing modules

### V2 Configuration Options
**Verbose Flag**: Control validation logging in RobustMPCController
- `config['verbose'] = False` (default): Silent operation for production
- `config['verbose'] = True`: Enable detailed validation logging for development
- Prevents log spam in high-frequency deployments

### Important File Locations
- **Central configuration**: `pyproject.toml` (unified package and dependency management)
- **Central environment**: `.venv/` (Python 3.12.3 with all versions and tools)
- **V1 data**: `V1/data/` (pre-processed datasets and trained models)
- **V2 library**: `V2/robust_mpc/` (production-ready modules)
- **V2 configuration**: `V2/config.yaml`, `V2/pyproject.toml`, `V2/Dockerfile`
- **V3 services**: `V3/services/` (microservice implementations)
- **Documentation**: Version-specific README files and DESIGN_DOCUMENT.md files

## Central Environment Setup

### Configuration Achievements
The project has been successfully migrated to a unified central environment with the following key improvements:

#### Enhanced Central Configuration (`pyproject.toml`)
- **Package discovery**: Added `V3.src.autopharm_core` to setuptools packages
- **Missing dependencies**: Added `joblib`, `scipy`, `typing-extensions`
- **License standardization**: Fixed format to PEP 639 standard (`CC-BY-NC-SA-4.0`)
- **Python version consistency**: Updated to Python 3.12 across all configurations
- **Tool harmonization**: Updated black (py312), mypy (3.12), coverage (all versions)
- **Test path expansion**: Includes `V1/tests`, `V2/tests`, `V3/tests`

#### License Format Resolution
- **V2 compatibility**: Fixed `V2/pyproject.toml` license format conflicts
- **Installation success**: Resolved central package installation issues
- **Standard compliance**: All configurations now follow PEP 639 licensing format

#### Module Discovery and Imports
- ✅ **V1.src modules**: `plant_simulator`, `model_architecture`, `mpc_controller`, `dataset`
- ✅ **V2.robust_mpc modules**: `core`, `models`, `estimators`, `optimizers`
- ✅ **V3.src.autopharm_core modules**: `control`, `learning`, `rl`, `xai`
- ✅ **Cross-version imports**: All inter-module dependencies working correctly

#### Development Tools Integration
- ✅ **Testing framework**: `pytest` across all versions with coverage reporting
- ✅ **Code formatting**: `black` and `isort` working on V1/, V2/, V3/
- ✅ **Type checking**: `mypy` configured for Python 3.12 across all versions
- ✅ **Dependency management**: `uv` with unified package installation

### Current Working Environment Status
```bash
# Environment verification (from project root)
source .venv/bin/activate
python -c "
import sys
print(f'Python: {sys.version}')

# Test all module imports
import V1.src.plant_simulator
import V2.robust_mpc.core  
import V3.src.autopharm_core
print('✅ All modules discoverable')

# Test cross-version imports
from V1.src import plant_simulator
from V2.robust_mpc import models
print('✅ Cross-version imports working')
"
```

### Migration Benefits
1. **Simplified workflow**: Single environment activation for all development tasks
2. **Consistent dependencies**: No version conflicts between V1, V2, V3 environments
3. **Improved testing**: Unified test execution across all project versions
4. **Enhanced CI/CD**: Single environment setup for automated workflows
5. **Developer experience**: Reduced context switching and setup complexity

### Technical Resolutions Achieved
During the central environment setup, several critical technical issues were resolved:

#### Package Configuration Issues
- **License classifier conflicts**: Resolved incompatible license formats between central and V2 configurations
- **Missing package discovery**: Fixed setuptools package finding for V3 autopharm_core modules
- **Dependency gaps**: Added missing dependencies (joblib, scipy, typing-extensions) that caused import failures
- **Python version mismatches**: Harmonized Python 3.12 requirements across all tool configurations

#### Installation and Import Problems
- **CLI script conflicts**: Removed problematic console_scripts that caused packaging issues
- **Module path resolution**: Fixed import paths for cross-version module dependencies
- **Package namespace conflicts**: Resolved setuptools package discovery conflicts
- **Environment isolation**: Ensured clean central environment without version-specific conflicts

#### Development Tool Integration
- **Test discovery**: Configured pytest to find tests across V1/, V2/, V3/ directories
- **Coverage reporting**: Updated coverage configuration to include all three versions
- **Code quality tools**: Harmonized black, mypy, isort configurations for consistent code style
- **Command execution**: Verified all development commands work from central environment

This central environment setup provides a robust foundation for continued development across all three versions of the PharmaControl system.

## Documentation Standards

### Professional Docstring Guidelines
All code modules follow comprehensive docstring standards:

- **Google/Sphinx style formatting** with clear sections for Args, Returns, Raises, Notes
- **Domain-specific context** integrating pharmaceutical process control terminology
- **Complete parameter documentation** including tensor shapes, units, and typical ranges
- **Usage examples** demonstrating practical implementation patterns
- **Technical implementation details** explaining algorithm choices and performance considerations
- **Control theory integration** connecting deep learning architecture to MPC requirements

### V1 Documentation Updates (Recently Enhanced)
- **MPC Controller** (`src/mpc_controller.py`): Comprehensive discrete optimization documentation
- **Plant Simulator** (`src/plant_simulator.py`): High-fidelity process dynamics with engineering context
- **Model Architecture** (`src/model_architecture.py`): Transformer implementation for pharmaceutical applications
- **Dataset Module** (`src/dataset.py`): Time series extraction for sequence-to-sequence learning
- **Package Init** (`src/__init__.py`): Professional package structure with dependency management

### Code Quality Improvements
Recent enhancements to V1 codebase include:

#### Critical Bug Fixes
- **Scaling inconsistency** in MPC cost calculation: Fixed comparison of scaled vs unscaled tensors
- **Hard-coded array indexing**: Replaced with dynamic configuration-based indexing for soft sensors
- **Unhandled None cases**: Added robust fallback when optimization fails to find valid solutions
- **Device placement errors**: Fixed GPU/CPU tensor device mismatches

#### Enhanced Error Handling
- **Constraint validation**: Graceful handling of infeasible control candidates
- **Configuration validation**: Clear error messages for missing process variables
- **Fallback mechanisms**: Safe operation when optimization encounters edge cases
- **Dependency checking**: Version compatibility validation with informative warnings

#### Professional Documentation
- **Complete API documentation**: All classes and methods with comprehensive docstrings
- **Domain expertise**: Pharmaceutical process control terminology and context
- **Usage examples**: Practical code snippets demonstrating typical workflows
- **Technical implementation details**: Algorithm explanations and performance considerations
- **Type safety**: Tensor shape specifications and validation throughout

## Development Philosophy

- **Safety first**: All control systems include constraint handling and safety limits
- **Uncertainty awareness**: V2+ explicitly models and uses prediction uncertainty
- **Industrial readiness**: Code follows production patterns with proper error handling
- **Educational value**: Clear documentation and progressive learning materials
- **Reproducible research**: Fixed random seeds and deterministic behavior where possible
- **Professional documentation**: Comprehensive docstrings following industry standards
- **Code quality**: Rigorous testing, error handling, and type safety practices

## Style Guide

### Code Output and Logging
- **Concise responses**: Keep explanations brief and technical
- **No decorative symbols**: Avoid excessive use of ✅🎯📊🔧 in production code
- **Professional logging**: Use clear, parseable log messages
- **Verbose flag**: Use `config['verbose']` to control detailed logging
- **Production-ready**: Default to minimal output suitable for enterprise deployment

### Response Style
- **Direct answers**: Address the specific question without elaboration
- **Technical focus**: Prioritize technical accuracy over descriptive language
- **Minimal preamble**: Avoid unnecessary introductions or conclusions
- **Actionable content**: Focus on what needs to be done