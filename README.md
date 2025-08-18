# PharmaControl: Advanced Process Control Evolution

[![V1: Prototype](https://img.shields.io/badge/V1-Prototype-blue.svg)](V1/README.md)
[![V2: Industrial](https://img.shields.io/badge/V2-Industrial%20Grade-green.svg)](V2/README.md)
[![V3: Autonomous](https://img.shields.io/badge/V3-Autonomous%20System-purple.svg)](V3/README.md)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-orange.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://python.org)

**From Prototype to Production to Autonomous Intelligence: The Complete Evolution of Advanced Process Control**

## 🚀 Project Overview

PharmaControl demonstrates the complete evolution from research prototype to autonomous industrial control system for pharmaceutical continuous granulation processes. This project showcases the integration of machine learning, control theory, uncertainty quantification, advanced optimization, reinforcement learning, and explainable AI in a real-world industrial context.

### 🎯 Three-Stage Evolution

1. **V1 (Prototype)**: Educational foundation with Transformer-based prediction and basic MPC
2. **V2 (Industrial)**: Production-ready system with uncertainty quantification and Kalman filtering  
3. **V3 (Autonomous)**: Self-learning system with online adaptation, RL, and explainable AI

## 🏗️ Unified Architecture

This project uses a **central environment setup** with unified dependency management across all versions:

```
PharmaControl/
├── 📋 README.md                     # This overview document
├── ⚙️ pyproject.toml                # Central configuration & dependencies
├── 🔧 .venv/                        # Unified virtual environment (Python 3.12)
├── 📁 V1/                           # Prototype system (Educational foundation)
│   ├── 📖 README.md                # V1 documentation
│   ├── ⚙️ pyproject.toml           # V1-specific configuration
│   ├── 📓 notebooks/               # 5-notebook educational series
│   ├── 🏗️ src/                     # V1 implementation modules
│   ├── 💾 data/                    # V1 datasets and models
│   └── 📚 docs/                    # V1 documentation (Sphinx)
├── 📁 V2/                          # Industrial-grade system (Production-ready)
│   ├── 📖 README.md                # V2 documentation  
│   ├── ⚙️ pyproject.toml           # V2-specific configuration
│   ├── 📓 notebooks/               # 5-notebook advanced series
│   ├── 🏭 robust_mpc/              # Industrial-grade library
│   ├── 🧪 tests/                   # Comprehensive test suite
│   └── 🚀 run_controller.py        # Production controller entry point
└── 📁 V3/                          # Autonomous system (AI-driven)
    ├── 📖 README.md                # V3 documentation
    ├── ⚙️ pyproject.toml           # V3-specific configuration  
    ├── 📓 notebooks/               # 3-notebook autonomous series
    ├── 🤖 services/                # Microservice architecture
    │   ├── control_agent/          # Main control orchestration
    │   ├── learning_service/       # Online model adaptation
    │   └── monitoring_xai_service/ # Explainable AI monitoring
    ├── 🧠 src/autopharm_core/      # Core autonomous components
    └── 🧪 tests/                   # Advanced testing suite
```

## 🛠️ Quick Setup (Unified Environment)

### Prerequisites
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager

### One-Command Setup
```bash
# Clone and setup entire project
git clone https://github.com/pharmacontrol/PharmaControl.git
cd PharmaControl

# Activate central environment
source .venv/bin/activate

# Install with all development tools
uv pip install -e ".[dev,notebooks]"

# Verify installation
python -c "
from V1.src import plant_simulator
from V2.robust_mpc import core  
from V3.src.autopharm_core.learning import online_trainer
print('✅ All versions accessible from central environment')
"
```

### Version-Specific Installation
```bash
# For V1 prototype work
uv pip install -e ".[v1,notebooks]"

# For V2 production deployment  
uv pip install -e ".[v2,production]"

# For V3 autonomous research
uv pip install -e ".[v3,rl-advanced,xai-research]"

# For complete development
uv pip install -e ".[full]"
```

## 🎯 Version Comparison

| Feature | V1 (Prototype) | V2 (Industrial) | V3 (Autonomous) | Evolution |
|---------|----------------|-----------------|-----------------|-----------|
| **Architecture** | Monolithic | Modular Library | Microservices | ✅ Scalable |
| **State Estimation** | Raw measurements | Kalman filtering | Adaptive filtering | ✅ Intelligence |
| **Uncertainty** | Point estimates | Probabilistic models | Bayesian online learning | ✅ Continuous improvement |
| **Optimization** | Grid search | Genetic algorithms | Reinforcement learning | ✅ Self-optimization |
| **Adaptation** | Static | Fixed parameters | Online learning | ✅ Autonomous evolution |
| **Explainability** | None | Basic metrics | XAI integration | ✅ Transparent decisions |
| **Deployment** | Local notebooks | Production server | Cloud microservices | ✅ Modern architecture |
| **Safety** | Basic constraints | Risk-adjusted costs | Learned safety policies | ✅ Adaptive safety |

## 🎓 Learning Path

### **🔵 Level 1: V1 Foundation Building**
Perfect for understanding core concepts:
- Process simulation and modeling  
- Machine learning for control
- Basic MPC implementation
- Performance analysis fundamentals

**⏱️ Time Investment**: 2-3 days  
👉 **[Start with V1 Documentation](V1/README.md)**

### **🟢 Level 2: V2 Industrial Implementation**
For production-ready, industrial-grade systems:
- State-space estimation and Kalman filtering
- Uncertainty quantification and probabilistic modeling
- Advanced optimization with evolutionary algorithms
- Integrated robust control with formal guarantees

**⏱️ Time Investment**: 1-2 weeks  
👉 **[Advance to V2 Documentation](V2/README.md)**

### **🟣 Level 3: V3 Autonomous Intelligence**
For cutting-edge autonomous control systems:
- Online learning and model adaptation
- Reinforcement learning for policy optimization
- Explainable AI for trust and transparency
- Microservice architecture for scalability

**⏱️ Time Investment**: 2-4 weeks  
👉 **[Master V3 Documentation](V3/README.md)**

## 🏭 Industrial Applications

This complete framework addresses real-world challenges across multiple industries:

### **Pharmaceutical Manufacturing**
- **V1**: Prototype granulation control
- **V2**: Production-grade tablet manufacturing
- **V3**: Autonomous quality assurance systems

### **Chemical Process Control**
- **V1**: Basic reactor control
- **V2**: Advanced distillation optimization  
- **V3**: Self-optimizing chemical plants

### **Advanced Materials**
- **V1**: Polymer processing fundamentals
- **V2**: Industrial composite manufacturing
- **V3**: Adaptive additive manufacturing

## 🔬 Technical Innovation Highlights

### **V1 Innovations (Educational Foundation)**
- **Transformer-based prediction**: State-of-the-art sequence modeling
- **Hybrid modeling**: Physics-informed soft sensors
- **Custom loss functions**: Horizon-weighted optimization
- **Realistic simulation**: Nonlinear dynamics with disturbances

### **V2 Innovations (Industrial Breakthrough)**
- **Uncertainty quantification**: Monte Carlo Dropout for confidence bounds
- **Risk-aware optimization**: Upper Confidence Bound approach
- **State estimation**: Kalman filtering for sensor noise rejection
- **Integral action**: Automatic disturbance learning and compensation
- **Evolutionary optimization**: Intelligent search through complex action spaces

### **V3 Innovations (Autonomous Intelligence)**
- **Online learning**: Real-time model adaptation from process data
- **Reinforcement learning**: Policy optimization through experience
- **Explainable AI**: LIME, SHAP, and custom interpretability methods
- **Microservice architecture**: Scalable, distributed control systems
- **Safety learning**: Adaptive constraint handling and risk assessment

## 🚀 Development Workflows

### **Code Quality & Testing**
```bash
# Code formatting and linting
black V1/ V2/ V3/
ruff V1/ V2/ V3/
isort V1/ V2/ V3/

# Type checking
mypy V1/ V2/ V3/

# Testing (from central environment)
pytest V2/tests/ -v                    # V2 industrial tests
pytest V3/tests/ -v                    # V3 autonomous tests

# Test coverage across all versions
pytest --cov=V1.src --cov=V2.robust_mpc --cov=V3.src.autopharm_core
```

### **Documentation**
```bash
# Build V1 documentation (Sphinx)
cd V1/docs && sphinx-build -b html . _build/html/

# Build V2 documentation (Sphinx)
cd V2/docs && sphinx-build -b html . _build/html/

# View documentation in browser
# Open V1/docs/_build/html/index.html or V2/docs/_build/html/index.html
```

### **Running Controllers**
```bash
# V1: Notebook-based execution
cd V1 && jupyter lab

# V2: Production controller
python V2/run_controller.py --config V2/config.yaml

# V3: Microservice deployment
cd V3 && docker-compose up  # (Future implementation)
```

## 📊 Educational Impact & Skills Development

### **Graduate-Level Concepts Covered**
- ✅ **Classical Control**: MPC, state estimation, robustness analysis
- ✅ **Machine Learning**: Transformers, uncertainty quantification, Bayesian methods
- ✅ **Advanced AI**: Reinforcement learning, online learning, transfer learning
- ✅ **Explainable AI**: Model interpretability, decision transparency, trust metrics
- ✅ **Modern Architecture**: Microservices, distributed systems, cloud deployment
- ✅ **Industrial Practice**: Process simulation, regulatory compliance, safety systems

### **Professional Skills Developed**
1. **V1 Skills**: Basic control system design, ML for control, Python proficiency
2. **V2 Skills**: Industrial software architecture, production deployment, testing
3. **V3 Skills**: Autonomous system design, microservices, explainable AI, cloud deployment

## 🔄 Central Environment Benefits

The unified environment approach provides:

- **✅ Consistent Dependencies**: All versions use same Python 3.12 and package versions
- **✅ Cross-Version Imports**: Easy comparison and component reuse between versions
- **✅ Unified Development**: Single environment for development across all versions
- **✅ Simplified Testing**: Integrated test suite covering all components
- **✅ Enhanced Tooling**: Shared code quality tools (black, ruff, mypy, pytest)

## 🤝 Contributing

This project welcomes contributions for:

- 📝 **Documentation**: Tutorials, examples, API documentation
- 🧪 **Testing**: Additional test cases, validation scenarios, benchmarks
- 🔧 **Features**: New algorithms, optimization methods, control strategies
- 🎨 **Visualization**: Analysis tools, dashboards, monitoring interfaces
- 🏭 **Applications**: New industrial case studies and implementations
- 🔬 **Research**: Novel AI/ML techniques for autonomous control

### Contribution Guidelines
```bash
# Setup development environment
uv pip install -e ".[dev,testing,cicd]"

# Pre-commit hooks
pre-commit install

# Run full test suite before submitting
pytest --cov=V1.src --cov=V2.robust_mpc --cov=V3.src.autopharm_core
```

## 📄 License & Usage

**License**: CC BY-NC-SA 4.0 (Creative Commons Attribution-NonCommercial-ShareAlike 4.0)

This project is provided for educational and research purposes. Commercial use requires explicit permission. Please ensure compliance with applicable regulations when adapting for industrial use.

## 🏆 Achievement Pathway

### **🥉 Bronze: Foundation Mastery**
Complete V1 → Understand core MPC and ML integration

### **🥈 Silver: Industrial Expertise** 
Complete V1 + V2 → Master production-ready control systems

### **🥇 Gold: Autonomous Intelligence**
Complete V1 + V2 + V3 → Expert in autonomous control with explainable AI

### **🏆 Platinum: Innovation Leader**
Contribute to the project → Shape the future of intelligent process control

## 🔮 Roadmap & Future Development

### **Current Status**
- ✅ **V1**: Complete and documented
- ✅ **V2**: Complete with comprehensive testing
- 🚧 **V3**: Core components implemented, advanced features in development

### **Upcoming Features**
- **V3.1**: Complete microservice architecture with Docker/Kubernetes
- **V3.2**: Advanced RL algorithms (PPO, SAC, TD3)
- **V3.3**: Federated learning across multiple plants
- **V3.4**: Digital twin integration and simulation-to-reality transfer

### **Research Directions**
- **Safety-Critical RL**: Formal verification of learned policies
- **Causal AI**: Understanding cause-effect relationships in process control
- **Quantum Control**: Quantum-enhanced optimization algorithms
- **Human-AI Collaboration**: Interactive control system design

---

**🎯 Mission**: Demonstrate the complete evolution from academic research through industrial implementation to autonomous intelligence in process control.

**🔬 Vision**: Bridge the gap between control theory and autonomous industrial reality with uncertainty-aware, adaptive, explainable, and safety-critical systems.

**⭐ Start Your Journey**: 
[V1 Prototype](V1/README.md) → [V2 Industrial](V2/README.md) → [V3 Autonomous](V3/README.md) → **Future Pioneer**!

## 🌟 Complete Navigation Guide

### **📖 Documentation Index**

| Version | Focus Area | README | Key Features | Learning Time |
|---------|------------|--------|--------------|---------------|
| **Central** | Project Overview | [README.md](README.md) | Unified environment, project structure | 30 minutes |
| **V1** | Educational Foundation | [V1/README.md](V1/README.md) | Transformer-based MPC, 5 notebooks | 2-3 days |
| **V2** | Industrial Production | [V2/README.md](V2/README.md) | Uncertainty quantification, genetic optimization | 1-2 weeks |
| **V3** | Autonomous Intelligence | [V3/README.md](V3/README.md) | RL, XAI, online learning, microservices | 2-4 weeks |

### **🔗 Cross-Reference Matrix**

| From → To | Purpose | Link Pattern |
|-----------|---------|--------------|
| **Central → V1** | Start learning journey | `[V1 Prototype](V1/README.md)` |
| **Central → V2** | Production deployment | `[V2 Industrial](V2/README.md)` |
| **Central → V3** | Autonomous research | `[V3 Autonomous](V3/README.md)` |
| **V1 → V2** | Evolution pathway | `[V2 Industrial →](../V2/README.md)` |
| **V2 → V3** | Advanced intelligence | `[V3 Autonomous →](../V3/README.md)` |
| **V1/V2/V3 → Central** | Full project context | `[← Main README](../README.md)` |

### **⚙️ Configuration Files**

| File | Purpose | Key Dependencies |
|------|---------|------------------|
| **pyproject.toml** | Central configuration | All versions, unified deps |
| **V1/pyproject.toml** | Prototype setup | Educational dependencies |
| **V2/pyproject.toml** | Production config | Industrial dependencies |
| **V3/pyproject.toml** | Autonomous setup | RL/XAI dependencies |
| **CLAUDE.md** | Developer guide | Development workflows |

### **🚀 Quick Start Paths**

| Goal | Path | Commands |
|------|------|----------|
| **Learn Fundamentals** | V1 Only | `cd V1 && jupyter lab` |
| **Deploy Production** | V1 → V2 | `python V2/run_controller.py` |
| **Research Autonomy** | V1 → V2 → V3 | `cd V3 && jupyter lab` |
| **Full Development** | Central Environment | `uv pip install -e ".[full]"` |