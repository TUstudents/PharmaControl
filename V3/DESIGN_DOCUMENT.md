# AutoPharm: V3 Design Document

**Version:** 3.0.0  
**Date:** 2025-01-15  
**Status:** Implementation Ready  

---

## Vision Statement

**"To create a fully autonomous, self-improving, and trustworthy control framework for complex manufacturing processes."**

AutoPharm V3 represents the evolutionary leap from industrial-grade control (V2) to truly autonomous manufacturing intelligence. This framework combines cutting-edge artificial intelligence, explainable decision-making, and continuous learning to create a control system that not only optimizes processes but also explains its reasoning and adapts to changing conditions without human intervention.

---

## Core Pillars

### 1. Online Learning & Adaptation
- **Continuous Model Updates**: Real-time model retraining as new data becomes available
- **Performance Monitoring**: Automatic detection of model degradation and drift
- **Adaptive Hyperparameters**: Dynamic adjustment of control parameters based on performance metrics
- **Transfer Learning**: Knowledge transfer between similar processes and operating conditions

### 2. Explainable AI & Trust
- **Decision Transparency**: Every control action comes with human-interpretable explanations
- **Uncertainty Communication**: Clear quantification and communication of prediction confidence
- **Audit Trail**: Complete logging of decisions, explanations, and performance metrics
- **Human-AI Collaboration**: Seamless integration with human operators and domain experts

### 3. Advanced Policy Learning (Reinforcement Learning)
- **Goal-Oriented Control**: Learning optimal policies for complex, multi-objective control problems
- **Safe Exploration**: Constraint-aware policy learning that respects process safety limits
- **Multi-Agent Coordination**: Distributed control across multiple process units
- **Hierarchical Decision Making**: Strategic planning combined with tactical control actions

---

## Technology Stack & Assumptions

### Core Technologies
- **Python**: 3.12+
- **Machine Learning**: PyTorch 2.0+, scikit-learn
- **Classical Control**: pykalman, DEAP (from V2)
- **Reinforcement Learning**: Stable-Baselines3
- **Explainable AI**: SHAP
- **Web Framework**: FastAPI, Uvicorn
- **Data Processing**: Pandas, NumPy
- **Testing**: pytest

### Infrastructure Assumptions
- **High-fidelity simulator**: `AdvancedPlantSimulator` available from V1/V2
- **Data infrastructure**: MQTT/Kafka for inter-service communication
- **Time-series database**: InfluxDB for logging and historical data
- **Container orchestration**: Docker/Kubernetes for deployment

### Made-Up Elements (Fictional but Realistic)
- **Granulation process model**: Mathematical relationships in simulator represent realistic pharmaceutical continuous granulation
- **Process constraints**: Equipment limits and operational ranges based on industry standards
- **Quality metrics**: d50 (particle size) and LOD (loss on drying) as representative pharmaceutical CMAs

---

## V3 Architecture: Multi-Service Design

AutoPharm V3 employs a **microservices architecture** with three core services:

### Control Agent (Real-time Service)
- **Primary Function**: Real-time process control and decision making
- **Response Time**: < 100ms for control decisions
- **Technology**: FastAPI, asyncio, real-time data processing
- **Responsibilities**:
  - Execute control policies in real-time
  - Maintain process safety and constraints
  - Interface with plant hardware and sensors
  - Provide immediate explanations for control actions

### Learning Service (Asynchronous Service)
- **Primary Function**: Continuous model training and improvement
- **Processing Mode**: Background, batch processing
- **Technology**: PyTorch, distributed training
- **Responsibilities**:
  - Online model retraining and updates
  - Performance monitoring and drift detection
  - Transfer learning across processes
  - Hyperparameter optimization

### Monitoring & XAI Service (API/Web Service)
- **Primary Function**: Explainability, visualization, and human interface
- **Interface**: REST API + Web Dashboard
- **Technology**: FastAPI, SHAP integration
- **Responsibilities**:
  - Generate explanations for control decisions
  - Visualize model performance and behavior
  - Provide human-interpretable insights
  - Manage audit trails and compliance reporting

---

## V3 File Structure and Public API Definition

This section defines the complete file structure and the public-facing API for every module within the `autopharm_core` library. **Every function signature, class definition, and data contract is specified to ensure implementation clarity.**

### Directory Structure
```
V3/
├── services/                           # Microservices
│   ├── control_agent/                 # Real-time control service
│   │   ├── Dockerfile
│   │   └── main.py
│   ├── learning_service/              # Machine learning service
│   │   ├── Dockerfile
│   │   └── main.py
│   └── monitoring_xai_service/        # Explainability service
│       ├── Dockerfile
│       └── main.py
│       
├── src/                               # Shared core libraries
│   └── autopharm_core/
│       ├── __init__.py
│       ├── common/
│       │   ├── __init__.py
│       │   └── types.py              # Type definitions and data contracts
│       ├── control/
│       │   ├── __init__.py
│       │   ├── estimators.py         # Kalman filtering (from V2)
│       │   ├── mpc.py                # Robust MPC (from V2)
│       │   └── optimizers.py         # Genetic algorithms (from V2)
│       ├── learning/
│       │   ├── __init__.py
│       │   ├── data_handler.py       # Database interface
│       │   ├── online_trainer.py     # Continuous learning
│       │   └── rl_policy.py          # Reinforcement learning
│       ├── models/
│       │   ├── __init__.py
│       │   └── probabilistic.py      # Enhanced transformer from V2
│       └── xai/
│           ├── __init__.py
│           └── explainer.py          # SHAP-based explanations
│           
├── tests/                             # Comprehensive test suite
│   ├── test_control/                 # Control system tests
│   ├── test_learning/                # Learning system tests
│   └── test_xai/                     # XAI tests
│   
├── config/                           # Configuration files
│   ├── control_agent.yaml          # Control parameters
│   └── learning_service.yaml       # Learning parameters
│   
├── data/                             # Data storage
│   ├── models/                       # Trained models
│   └── scalers/                      # Data preprocessors
│   
├── .github/                          # GitHub workflows
│   └── workflows/
│       └── python-ci.yml           # Continuous integration
│       
├── docker-compose.yml                # Multi-service deployment
├── pyproject.toml                    # Project configuration
├── requirements.txt                  # Dependencies
└── README.md                         # Project overview
```

---

## Library API: `autopharm_core`

### **1. Module: `common/types.py`**
**Purpose:** Defines shared data structures and types for clear contracts between modules.

```python
from pydantic import BaseModel
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import numpy as np
import torch

class StateVector(BaseModel):
    """Represents a single timestep of process state"""
    timestamp: float
    cmas: Dict[str, float]  # Critical Material Attributes: {'d50': 380.0, 'lod': 1.8}
    cpps: Dict[str, float]  # Critical Process Parameters: {'spray_rate': 120.0, 'air_flow': 500.0, 'carousel_speed': 30.0}
    
    class Config:
        arbitrary_types_allowed = True

class ControlAction(BaseModel):
    """Control action with metadata"""
    timestamp: float
    cpp_setpoints: Dict[str, float]  # New setpoints to apply
    action_id: str                   # Unique identifier for this action
    confidence: float               # Confidence score [0.0, 1.0]
    
    class Config:
        arbitrary_types_allowed = True

class ModelPrediction(BaseModel):
    """Probabilistic model prediction with uncertainty"""
    mean: np.ndarray        # Shape: (horizon, n_cma_features)
    std: np.ndarray         # Shape: (horizon, n_cma_features) 
    horizon: int           # Number of future timesteps predicted
    feature_names: List[str]  # Names of CMA features predicted
    
    class Config:
        arbitrary_types_allowed = True

class TrainingMetrics(BaseModel):
    """Model training performance metrics"""
    model_version: str
    validation_loss: float
    training_duration_seconds: float
    dataset_size: int
    hyperparameters: Dict[str, Any]
    
class DecisionExplanation(BaseModel):
    """Human-interpretable explanation for a control decision"""
    decision_id: str
    control_action: ControlAction
    narrative: str                    # Human-readable explanation
    feature_attributions: Dict[str, float]  # Feature name -> SHAP value
    confidence_factors: Dict[str, float]    # Factors affecting confidence
    alternatives_considered: int       # Number of alternative actions evaluated
```

### **2. Module: `models/probabilistic.py`**
**Purpose:** Contains the core predictive model architectures (enhanced V2 Transformer).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any
import numpy as np

class ProbabilisticTransformer(nn.Module):
    """
    Enhanced Transformer model from V2 with additional probabilistic capabilities
    for uncertainty quantification and improved control performance.
    """
    
    def __init__(self, 
                 cma_features: int, 
                 cpp_features: int, 
                 d_model: int = 128,
                 nhead: int = 8,
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 2,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        """
        Initialize the Transformer model architecture.
        
        Args:
            cma_features: Number of Critical Material Attributes (CMAs)
            cpp_features: Number of Critical Process Parameters (CPPs) including soft sensors
            d_model: Model dimension for transformer
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
        """
        super().__init__()
        self.cma_features = cma_features
        self.cpp_features = cpp_features
        self.d_model = d_model
        self.dropout = dropout
        
        # Input projections
        self.cma_projection = nn.Linear(cma_features, d_model)
        self.cpp_projection = nn.Linear(cpp_features, d_model)
        
        # Transformer components
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers
        self.output_projection = nn.Linear(d_model, cma_features)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, 
                past_cmas: torch.Tensor, 
                past_cpps: torch.Tensor, 
                future_cpps: torch.Tensor) -> torch.Tensor:
        """
        Standard deterministic forward pass.
        
        Args:
            past_cmas: Historical CMAs, shape (batch_size, lookback, cma_features)
            past_cpps: Historical CPPs, shape (batch_size, lookback, cpp_features)  
            future_cpps: Future CPPs, shape (batch_size, horizon, cpp_features)
            
        Returns:
            torch.Tensor: Predicted future CMAs, shape (batch_size, horizon, cma_features)
        """
        batch_size, lookback, _ = past_cmas.shape
        _, horizon, _ = future_cpps.shape
        
        # Project inputs to model dimension
        past_cmas_proj = self.cma_projection(past_cmas)
        past_cpps_proj = self.cpp_projection(past_cpps)
        future_cpps_proj = self.cpp_projection(future_cpps)
        
        # Combine past information for encoder
        encoder_input = past_cmas_proj + past_cpps_proj
        
        # Use future CPPs as decoder input
        decoder_input = future_cpps_proj
        
        # Apply transformer
        transformer_output = self.transformer(
            src=encoder_input,
            tgt=decoder_input
        )
        
        # Project to output space
        output = self.output_projection(transformer_output)
        
        return output
        
    def predict_distribution(self, 
                           past_cmas: torch.Tensor, 
                           past_cpps: torch.Tensor, 
                           future_cpps: torch.Tensor, 
                           n_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs Monte Carlo Dropout to get predictive distribution.
        
        Args:
            past_cmas: Historical CMAs, shape (batch_size, lookback, cma_features)
            past_cpps: Historical CPPs, shape (batch_size, lookback, cpp_features)
            future_cpps: Future CPPs, shape (batch_size, horizon, cpp_features)
            n_samples: Number of MC dropout samples
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (mean_prediction, std_prediction)
                Both have shape (batch_size, horizon, cma_features)
        """
        self.train()  # Enable dropout for MC sampling
        
        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.forward(past_cmas, past_cpps, future_cpps)
                predictions.append(pred)
                
        predictions = torch.stack(predictions)  # (n_samples, batch_size, horizon, cma_features)
        
        mean_prediction = predictions.mean(dim=0)
        std_prediction = predictions.std(dim=0)
        
        self.eval()  # Return to evaluation mode
        
        return mean_prediction, std_prediction
        
    def get_model_info(self) -> Dict[str, Any]:
        """Returns model configuration and parameter count."""
        return {
            'architecture': 'ProbabilisticTransformer',
            'cma_features': self.cma_features,
            'cpp_features': self.cpp_features,
            'd_model': self.d_model,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
```

### **3. Module: `control/estimators.py`**
**Purpose:** Kalman filtering and state estimation (enhanced from V2).

```python
import numpy as np
from pykalman import KalmanFilter
from typing import Optional, Tuple, Dict, Any
from ..common.types import StateVector

class KalmanStateEstimator:
    """
    Enhanced Kalman filter for state estimation with process control applications.
    Provides filtered state estimates and uncertainty quantification.
    """
    
    def __init__(self, 
                 transition_matrix: np.ndarray,
                 control_matrix: np.ndarray, 
                 initial_state_mean: np.ndarray,
                 process_noise_std: float = 1.0,
                 measurement_noise_std: float = 15.0):
        """
        Initialize the Kalman Filter.
        
        Args:
            transition_matrix: State transition matrix A
            control_matrix: Control input matrix B
            initial_state_mean: Initial state estimate
            process_noise_std: Process noise standard deviation
            measurement_noise_std: Measurement noise standard deviation
        """
        self.n_states = len(initial_state_mean)
        self.control_matrix = control_matrix
        
        # Initialize Kalman filter
        self.kf = KalmanFilter(
            transition_matrices=transition_matrix,
            observation_matrices=np.eye(self.n_states),
            transition_covariance=np.eye(self.n_states) * (process_noise_std ** 2),
            observation_covariance=np.eye(self.n_states) * (measurement_noise_std ** 2),
            initial_state_mean=initial_state_mean,
            initial_state_covariance=np.eye(self.n_states)
        )
        
        # State tracking
        self.filtered_state_mean = initial_state_mean
        self.filtered_state_covariance = np.eye(self.n_states)
        self.measurement_history = []
        self.control_history = []
        
    def estimate(self, measurement: np.ndarray, control_input: np.ndarray) -> np.ndarray:
        """
        Perform one Kalman filter update step.
        
        Args:
            measurement: Current measurement vector
            control_input: Current control input vector
            
        Returns:
            np.ndarray: Filtered state estimate
        """
        # Store history
        self.measurement_history.append(measurement.copy())
        self.control_history.append(control_input.copy())
        
        # Apply control input to transition
        transition_offset = np.dot(self.control_matrix, control_input)
        
        # Perform Kalman filter update
        self.filtered_state_mean, self.filtered_state_covariance = self.kf.filter_update(
            filtered_state_mean=self.filtered_state_mean,
            filtered_state_covariance=self.filtered_state_covariance,
            observation=measurement,
            transition_offset=transition_offset
        )
        
        return self.filtered_state_mean.copy()
    
    def get_uncertainty(self) -> np.ndarray:
        """
        Get current state uncertainty (standard deviations).
        
        Returns:
            np.ndarray: State uncertainty (diagonal of covariance matrix)
        """
        return np.sqrt(np.diag(self.filtered_state_covariance))
    
    def get_prediction(self, steps_ahead: int, future_controls: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict future states.
        
        Args:
            steps_ahead: Number of steps to predict
            future_controls: Future control inputs, shape (steps_ahead, n_controls)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (predicted_means, predicted_stds)
        """
        if future_controls is None:
            future_controls = np.zeros((steps_ahead, self.control_matrix.shape[1]))
            
        # Use Kalman filter's built-in prediction
        predicted_means, predicted_covariances = self.kf.smooth(
            X=np.array(self.measurement_history[-10:])  # Use recent history
        )
        
        # Extract uncertainties
        predicted_stds = np.array([np.sqrt(np.diag(cov)) for cov in predicted_covariances])
        
        return predicted_means, predicted_stds
    
    def reset(self, initial_state: np.ndarray):
        """Reset the filter to a new initial state."""
        self.filtered_state_mean = initial_state
        self.filtered_state_covariance = np.eye(self.n_states)
        self.measurement_history.clear()
        self.control_history.clear()
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get filter performance metrics."""
        if len(self.measurement_history) < 2:
            return {'innovation_variance': 0.0, 'filter_steps': 0}
            
        # Calculate innovation (prediction error) statistics
        innovations = []
        for i in range(1, len(self.measurement_history)):
            innovation = self.measurement_history[i] - self.measurement_history[i-1]
            innovations.append(np.linalg.norm(innovation))
            
        return {
            'innovation_variance': np.var(innovations) if innovations else 0.0,
            'average_uncertainty': np.mean(self.get_uncertainty()),
            'filter_steps': len(self.measurement_history)
        }
```

### **4. Module: `control/optimizers.py`**
**Purpose:** Genetic algorithm optimization (enhanced from V2).

```python
import numpy as np
from typing import Callable, List, Tuple, Dict, Any, Optional
from deap import base, creator, tools, algorithms
import random

class GeneticOptimizer:
    """
    Enhanced genetic algorithm optimizer for MPC control optimization.
    Uses DEAP library for robust evolutionary optimization.
    """
    
    def __init__(self, 
                 fitness_function: Callable[[np.ndarray], float],
                 param_bounds: List[Tuple[float, float]], 
                 ga_config: Dict[str, Any]):
        """
        Initialize the Genetic Algorithm optimizer.
        
        Args:
            fitness_function: Function that takes control plan and returns cost
            param_bounds: List of (min, max) bounds for each parameter
            ga_config: GA configuration with keys:
                - population_size: Population size
                - generations: Number of generations
                - mutation_rate: Mutation probability
                - crossover_rate: Crossover probability
                - tournament_size: Tournament selection size
        """
        self.fitness_function = fitness_function
        self.param_bounds = param_bounds
        self.config = ga_config
        
        # DEAP setup
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", self._rand_float_in_bounds)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, 
                             self.toolbox.attr_float, len(param_bounds))
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Genetic operators
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self._custom_mutate, 
                             indpb=ga_config.get('mutation_rate', 0.1))
        self.toolbox.register("select", tools.selTournament, 
                             tournsize=ga_config.get('tournament_size', 3))
        self.toolbox.register("evaluate", self._evaluate_individual)
        
    def _rand_float_in_bounds(self) -> float:
        """Generate random float within parameter bounds."""
        idx = random.randint(0, len(self.param_bounds) - 1)
        min_val, max_val = self.param_bounds[idx]
        return random.uniform(min_val, max_val)
    
    def _custom_mutate(self, individual, indpb: float):
        """Custom mutation that respects parameter bounds."""
        for i in range(len(individual)):
            if random.random() < indpb:
                min_val, max_val = self.param_bounds[i]
                individual[i] = random.uniform(min_val, max_val)
        return individual,
    
    def _evaluate_individual(self, individual: List[float]) -> Tuple[float,]:
        """Evaluate fitness of an individual."""
        try:
            fitness = self.fitness_function(np.array(individual))
            return (fitness,)
        except Exception as e:
            # Return high penalty for invalid individuals
            return (1e6,)
    
    def optimize(self, 
                 population_size: Optional[int] = None,
                 generations: Optional[int] = None) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
        Run the evolutionary algorithm and return the best solution.
        
        Args:
            population_size: Override default population size
            generations: Override default number of generations
            
        Returns:
            Tuple[np.ndarray, float, Dict[str, Any]]: 
                (best_solution, best_fitness, optimization_stats)
        """
        pop_size = population_size or self.config.get('population_size', 50)
        num_gen = generations or self.config.get('generations', 20)
        
        # Create initial population
        population = self.toolbox.population(n=pop_size)
        
        # Statistics tracking
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Run evolution
        population, logbook = algorithms.eaSimple(
            population, self.toolbox,
            cxpb=self.config.get('crossover_rate', 0.7),
            mutpb=self.config.get('mutation_rate', 0.1),
            ngen=num_gen,
            stats=stats,
            verbose=False
        )
        
        # Extract best solution
        best_individual = tools.selBest(population, 1)[0]
        best_fitness = best_individual.fitness.values[0]
        best_solution = np.array(best_individual)
        
        # Optimization statistics
        optimization_stats = {
            'generations_run': num_gen,
            'population_size': pop_size,
            'best_fitness': best_fitness,
            'convergence_history': [entry['min'] for entry in logbook],
            'final_population_diversity': self._calculate_diversity(population)
        }
        
        return best_solution, best_fitness, optimization_stats
    
    def _calculate_diversity(self, population: List) -> float:
        """Calculate population diversity metric."""
        if len(population) < 2:
            return 0.0
            
        distances = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                dist = np.linalg.norm(np.array(population[i]) - np.array(population[j]))
                distances.append(dist)
                
        return np.mean(distances) if distances else 0.0
```

### **5. Module: `control/mpc.py`**
**Purpose:** Robust MPC controller integrating all control components.

```python
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from ..common.types import StateVector, ControlAction, ModelPrediction
from ..models.probabilistic import ProbabilisticTransformer
from .estimators import KalmanStateEstimator
from .optimizers import GeneticOptimizer
import torch
import uuid
from datetime import datetime

class RobustMPCController:
    """
    Enhanced Model Predictive Controller integrating probabilistic prediction,
    state estimation, and evolutionary optimization for robust process control.
    """
    
    def __init__(self, 
                 model: ProbabilisticTransformer,
                 estimator: KalmanStateEstimator, 
                 optimizer_class: type,
                 config: Dict[str, Any],
                 scalers: Dict[str, Any]):
        """
        Initialize the Robust MPC controller.
        
        Args:
            model: Trained probabilistic transformer model
            estimator: Kalman state estimator
            optimizer_class: Genetic optimizer class
            config: MPC configuration dictionary
            scalers: Data scaling objects for normalization
        """
        self.model = model
        self.estimator = estimator
        self.optimizer_class = optimizer_class
        self.config = config
        self.scalers = scalers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Control performance tracking
        self.control_history = []
        self.performance_metrics = {}
        self.integral_error = np.zeros(len(config['cma_names']))
        
    def get_action(self, 
                   history: List[StateVector], 
                   setpoint: Dict[str, float],
                   use_uncertainty: bool = True) -> ControlAction:
        """
        Main public method for generating optimal control actions.
        
        Args:
            history: List of recent StateVector observations
            setpoint: Target setpoints for CMAs
            use_uncertainty: Whether to use uncertainty-aware optimization
            
        Returns:
            ControlAction: Optimal control action with metadata
        """
        if len(history) < self.config['lookback']:
            raise ValueError(f"Need at least {self.config['lookback']} historical points")
        
        # Extract recent history
        recent_history = history[-self.config['lookback']:]
        
        # Convert to structured data
        historical_data = self._prepare_historical_data(recent_history)
        
        # Apply state estimation
        filtered_state = self._apply_state_filtering(historical_data)
        
        # Create fitness function for optimization
        fitness_function = self._create_fitness_function(
            historical_data, setpoint, use_uncertainty
        )
        
        # Set up optimization bounds
        bounds = self._get_optimization_bounds(recent_history[-1])
        
        # Initialize genetic optimizer
        ga_config = {
            'population_size': self.config.get('population_size', 40),
            'generations': self.config.get('generations', 15),
            'mutation_rate': self.config.get('mutation_rate', 0.1),
            'crossover_rate': self.config.get('crossover_rate', 0.7),
            'tournament_size': self.config.get('tournament_size', 3)
        }
        
        optimizer = self.optimizer_class(fitness_function, bounds, ga_config)
        
        # Run optimization
        best_solution, best_fitness, opt_stats = optimizer.optimize()
        
        # Extract first control action from optimal sequence
        action_dict = self._decode_action(best_solution, recent_history[-1])
        
        # Calculate confidence based on optimization convergence and uncertainty
        confidence = self._calculate_confidence(opt_stats, best_fitness)
        
        # Create control action object
        control_action = ControlAction(
            timestamp=datetime.now().timestamp(),
            cpp_setpoints=action_dict,
            action_id=str(uuid.uuid4()),
            confidence=confidence
        )
        
        # Update control history
        self.control_history.append({
            'action': control_action,
            'setpoint': setpoint.copy(),
            'fitness': best_fitness,
            'optimization_stats': opt_stats
        })
        
        # Update integral error for offset-free control
        self._update_integral_error(recent_history[-1], setpoint)
        
        return control_action
    
    def _prepare_historical_data(self, history: List[StateVector]) -> Dict[str, np.ndarray]:
        """Convert StateVector list to arrays for model input."""
        cma_data = []
        cpp_data = []
        
        for state in history:
            # Extract CMAs
            cma_values = [state.cmas[name] for name in self.config['cma_names']]
            cma_data.append(cma_values)
            
            # Extract CPPs with soft sensors
            cpp_values = []
            for name in self.config['cpp_names_and_soft_sensors']:
                if name in state.cpps:
                    cpp_values.append(state.cpps[name])
                elif name == 'specific_energy':
                    # Calculate soft sensor
                    se = (state.cpps['spray_rate'] * state.cpps['carousel_speed']) / 1000.0
                    cpp_values.append(se)
                elif name == 'froude_number_proxy':
                    # Calculate soft sensor
                    fn = (state.cpps['carousel_speed']**2) / 9.81
                    cpp_values.append(fn)
                else:
                    cpp_values.append(0.0)  # Default value
            
            cpp_data.append(cpp_values)
        
        return {
            'cmas': np.array(cma_data),
            'cpps': np.array(cpp_data)
        }
    
    def _apply_state_filtering(self, historical_data: Dict[str, np.ndarray]) -> np.ndarray:
        """Apply Kalman filtering to recent measurements."""
        # Use the last measurement for filtering
        last_measurement = historical_data['cmas'][-1]
        last_control = historical_data['cpps'][-1][:len(self.config['cpp_names'])]
        
        filtered_state = self.estimator.estimate(last_measurement, last_control)
        return filtered_state
    
    def _create_fitness_function(self, 
                               historical_data: Dict[str, np.ndarray],
                               setpoint: Dict[str, float],
                               use_uncertainty: bool) -> Callable[[np.ndarray], float]:
        """Create fitness function for the genetic algorithm."""
        
        def fitness_function(action_sequence: np.ndarray) -> float:
            try:
                # Reshape to (horizon, n_cpps)
                action_matrix = action_sequence.reshape(self.config['horizon'], 
                                                      len(self.config['cpp_names']))
                
                # Add soft sensors to actions
                full_actions = self._add_soft_sensors_to_actions(action_matrix)
                
                # Scale data for model
                scaled_cmas = self._scale_data(historical_data['cmas'], 'cmas')
                scaled_cpps = self._scale_data(historical_data['cpps'], 'cpps')
                scaled_actions = self._scale_data(full_actions, 'cpps')
                
                # Convert to tensors
                past_cmas_tensor = torch.tensor(scaled_cmas, dtype=torch.float32).unsqueeze(0).to(self.device)
                past_cpps_tensor = torch.tensor(scaled_cpps, dtype=torch.float32).unsqueeze(0).to(self.device)
                future_cpps_tensor = torch.tensor(scaled_actions, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # Get model prediction
                if use_uncertainty:
                    mean_pred, std_pred = self.model.predict_distribution(
                        past_cmas_tensor, past_cpps_tensor, future_cpps_tensor, n_samples=25
                    )
                    # Risk-aware prediction: mean + beta * std
                    risk_beta = self.config.get('risk_beta', 1.0)
                    prediction = mean_pred + risk_beta * std_pred
                else:
                    prediction = self.model(past_cmas_tensor, past_cpps_tensor, future_cpps_tensor)
                
                # Unscale prediction
                unscaled_prediction = self._unscale_data(prediction.squeeze(0).detach().cpu().numpy(), 'cmas')
                
                # Calculate tracking error
                setpoint_array = np.array([setpoint[name] for name in self.config['cma_names']])
                setpoint_matrix = np.tile(setpoint_array, (self.config['horizon'], 1))
                
                tracking_error = np.mean(np.abs(unscaled_prediction - setpoint_matrix))
                
                # Add integral action for offset-free control
                integral_correction = self.config.get('integral_gain', 0.05) * np.linalg.norm(self.integral_error)
                
                # Control effort penalty
                control_effort = np.mean(np.diff(action_matrix, axis=0)**2) if action_matrix.shape[0] > 1 else 0.0
                control_penalty = self.config.get('control_effort_lambda', 0.05) * control_effort
                
                # Total cost
                total_cost = tracking_error + integral_correction + control_penalty
                
                return float(total_cost)
                
            except Exception as e:
                # Return high penalty for invalid solutions
                return 1e6
        
        return fitness_function
    
    def _get_optimization_bounds(self, last_state: StateVector) -> List[Tuple[float, float]]:
        """Get optimization bounds for control variables."""
        bounds = []
        current_cpps = [last_state.cpps[name] for name in self.config['cpp_names']]
        
        for i, name in enumerate(self.config['cpp_names']):
            constraints = self.config['process_constraints'][name]
            
            # Apply rate constraints
            max_change = constraints['max_change_per_step']
            current_val = current_cpps[i]
            
            min_bound = max(constraints['min_val'], current_val - max_change)
            max_bound = min(constraints['max_val'], current_val + max_change)
            
            # Replicate for entire horizon
            for _ in range(self.config['horizon']):
                bounds.append((min_bound, max_bound))
        
        return bounds
    
    def _add_soft_sensors_to_actions(self, actions: np.ndarray) -> np.ndarray:
        """Add soft sensor calculations to control actions."""
        n_steps, n_cpps = actions.shape
        n_soft_sensors = len(self.config['cpp_names_and_soft_sensors']) - n_cpps
        
        full_actions = np.zeros((n_steps, n_cpps + n_soft_sensors))
        full_actions[:, :n_cpps] = actions
        
        # Calculate soft sensors
        spray_rate = actions[:, 0]  # Assuming spray_rate is first
        carousel_speed = actions[:, 2]  # Assuming carousel_speed is third
        
        specific_energy = (spray_rate * carousel_speed) / 1000.0
        froude_number_proxy = (carousel_speed**2) / 9.81
        
        full_actions[:, n_cpps] = specific_energy
        full_actions[:, n_cpps + 1] = froude_number_proxy
        
        return full_actions
    
    def _scale_data(self, data: np.ndarray, data_type: str) -> np.ndarray:
        """Scale data using saved scalers."""
        scaled_data = np.zeros_like(data)
        
        if data_type == 'cmas':
            feature_names = self.config['cma_names']
        else:  # cpps
            feature_names = self.config['cpp_names_and_soft_sensors']
        
        for i, name in enumerate(feature_names):
            if name in self.scalers:
                scaled_data[:, i] = self.scalers[name].transform(
                    data[:, i].reshape(-1, 1)
                ).flatten()
            else:
                scaled_data[:, i] = data[:, i]  # No scaling available
        
        return scaled_data
    
    def _unscale_data(self, data: np.ndarray, data_type: str) -> np.ndarray:
        """Unscale data using saved scalers."""
        unscaled_data = np.zeros_like(data)
        
        if data_type == 'cmas':
            feature_names = self.config['cma_names']
        else:  # cpps
            feature_names = self.config['cpp_names_and_soft_sensors']
        
        for i, name in enumerate(feature_names):
            if name in self.scalers:
                unscaled_data[:, i] = self.scalers[name].inverse_transform(
                    data[:, i].reshape(-1, 1)
                ).flatten()
            else:
                unscaled_data[:, i] = data[:, i]  # No scaling available
        
        return unscaled_data
    
    def _decode_action(self, solution: np.ndarray, last_state: StateVector) -> Dict[str, float]:
        """Extract first control action from optimization solution."""
        action_matrix = solution.reshape(self.config['horizon'], len(self.config['cpp_names']))
        first_action = action_matrix[0]
        
        return dict(zip(self.config['cpp_names'], first_action))
    
    def _calculate_confidence(self, opt_stats: Dict[str, Any], fitness: float) -> float:
        """Calculate confidence based on optimization quality and convergence."""
        # Base confidence on optimization convergence
        convergence_history = opt_stats['convergence_history']
        if len(convergence_history) > 1:
            improvement = (convergence_history[0] - convergence_history[-1]) / convergence_history[0]
            convergence_confidence = min(1.0, improvement * 2)  # Scale to [0, 1]
        else:
            convergence_confidence = 0.5
        
        # Factor in population diversity (higher diversity = more exploration)
        diversity = opt_stats.get('final_population_diversity', 0.0)
        diversity_confidence = min(1.0, diversity / 10.0)  # Normalize by expected range
        
        # Combine factors
        overall_confidence = 0.7 * convergence_confidence + 0.3 * diversity_confidence
        return max(0.1, min(0.95, overall_confidence))  # Clamp to reasonable range
    
    def _update_integral_error(self, current_state: StateVector, setpoint: Dict[str, float]):
        """Update integral error for offset-free control."""
        current_cmas = np.array([current_state.cmas[name] for name in self.config['cma_names']])
        setpoint_array = np.array([setpoint[name] for name in self.config['cma_names']])
        
        error = setpoint_array - current_cmas
        self.integral_error += self.config.get('integral_gain', 0.05) * error
        
        # Anti-windup: clamp integral error
        max_integral = self.config.get('max_integral_error', 100.0)
        self.integral_error = np.clip(self.integral_error, -max_integral, max_integral)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive controller performance metrics."""
        if not self.control_history:
            return {'message': 'No control history available'}
        
        recent_actions = self.control_history[-20:]  # Last 20 actions
        
        # Calculate performance metrics
        fitness_values = [entry['fitness'] for entry in recent_actions]
        confidences = [entry['action'].confidence for entry in recent_actions]
        
        metrics = {
            'average_fitness': np.mean(fitness_values),
            'fitness_trend': np.polyfit(range(len(fitness_values)), fitness_values, 1)[0],
            'average_confidence': np.mean(confidences),
            'control_consistency': 1.0 - np.std(fitness_values) / (np.mean(fitness_values) + 1e-6),
            'total_actions': len(self.control_history),
            'integral_error_norm': np.linalg.norm(self.integral_error),
            'estimator_metrics': self.estimator.get_performance_metrics()
        }
        
        return metrics
    
    def reset_controller(self):
        """Reset controller state for new operation."""
        self.control_history.clear()
        self.integral_error = np.zeros(len(self.config['cma_names']))
        # Note: Estimator reset should be done separately if needed
```

### **6. Module: `learning/data_handler.py`**
**Purpose:** Interface with time-series database for continuous learning.

```python
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from ..common.types import StateVector, TrainingMetrics
import sqlite3  # Simplified DB for demo (replace with InfluxDB in production)

class DataHandler:
    """
    Handles data storage, retrieval, and preprocessing for online learning.
    Interfaces with time-series database to manage operational data.
    """
    
    def __init__(self, db_connection_string: str):
        """
        Initialize database connection.
        
        Args:
            db_connection_string: Database connection string or path
        """
        self.db_path = db_connection_string
        self._initialize_database()
        
    def _initialize_database(self):
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            # Process data table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS process_data (
                    timestamp REAL PRIMARY KEY,
                    d50 REAL,
                    lod REAL,
                    spray_rate REAL,
                    air_flow REAL,
                    carousel_speed REAL,
                    specific_energy REAL,
                    froude_number_proxy REAL
                )
            """)
            
            # Model performance table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    timestamp REAL,
                    model_version TEXT,
                    validation_loss REAL,
                    dataset_size INTEGER,
                    training_duration REAL
                )
            """)
            
            conn.commit()
    
    def log_trajectory(self, trajectory: List[StateVector]):
        """
        Log a completed trajectory to the database.
        
        Args:
            trajectory: List of StateVector observations
        """
        with sqlite3.connect(self.db_path) as conn:
            for state in trajectory:
                # Prepare data row
                data_row = {
                    'timestamp': state.timestamp,
                    'd50': state.cmas.get('d50', 0.0),
                    'lod': state.cmas.get('lod', 0.0),
                    'spray_rate': state.cpps.get('spray_rate', 0.0),
                    'air_flow': state.cpps.get('air_flow', 0.0),
                    'carousel_speed': state.cpps.get('carousel_speed', 0.0),
                    'specific_energy': state.cpps.get('specific_energy', 0.0),
                    'froude_number_proxy': state.cpps.get('froude_number_proxy', 0.0)
                }
                
                # Insert with conflict resolution
                conn.execute("""
                    INSERT OR REPLACE INTO process_data 
                    (timestamp, d50, lod, spray_rate, air_flow, carousel_speed, 
                     specific_energy, froude_number_proxy)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, tuple(data_row.values()))
            
            conn.commit()
    
    def fetch_recent_data(self, duration_hours: int) -> pd.DataFrame:
        """
        Fetch recent operational data for retraining.
        
        Args:
            duration_hours: Number of hours of recent data to fetch
            
        Returns:
            pd.DataFrame: Recent process data
        """
        end_time = datetime.now().timestamp()
        start_time = end_time - (duration_hours * 3600)
        
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT * FROM process_data 
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp
            """
            
            df = pd.read_sql_query(query, conn, params=(start_time, end_time))
        
        return df
    
    def fetch_training_data(self, 
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None,
                          min_samples: int = 1000) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Fetch data suitable for model training with preprocessing.
        
        Args:
            start_time: Start time for data range
            end_time: End time for data range
            min_samples: Minimum number of samples required
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: (training_data, metadata)
        """
        # Set default time range if not provided
        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            start_time = end_time - timedelta(days=7)  # Last week by default
        
        start_ts = start_time.timestamp()
        end_ts = end_time.timestamp()
        
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT * FROM process_data 
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp
            """
            
            df = pd.read_sql_query(query, conn, params=(start_ts, end_ts))
        
        # Data quality checks
        metadata = {
            'raw_samples': len(df),
            'time_range': (start_time, end_time),
            'data_quality': self._assess_data_quality(df)
        }
        
        # Basic preprocessing
        if len(df) >= min_samples:
            # Remove duplicates and sort
            df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            
            # Fill missing values with interpolation
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].interpolate(method='linear')
            
            # Remove remaining NaN rows
            df = df.dropna()
            
            metadata['processed_samples'] = len(df)
            metadata['preprocessing_applied'] = ['deduplication', 'interpolation', 'nan_removal']
        else:
            metadata['warning'] = f'Insufficient data: {len(df)} < {min_samples}'
        
        return df, metadata
    
    def log_training_metrics(self, metrics: TrainingMetrics):
        """
        Log model training metrics to the database.
        
        Args:
            metrics: Training metrics to log
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO model_performance 
                (timestamp, model_version, validation_loss, dataset_size, training_duration)
                VALUES (?, ?, ?, ?, ?)
            """, (
                datetime.now().timestamp(),
                metrics.model_version,
                metrics.validation_loss,
                metrics.dataset_size,
                metrics.training_duration_seconds
            ))
            conn.commit()
    
    def get_training_history(self, 
                           model_version: Optional[str] = None,
                           limit: int = 100) -> pd.DataFrame:
        """
        Get historical training performance data.
        
        Args:
            model_version: Specific model version to filter (optional)
            limit: Maximum number of records to return
            
        Returns:
            pd.DataFrame: Training history
        """
        with sqlite3.connect(self.db_path) as conn:
            if model_version:
                query = """
                    SELECT * FROM model_performance 
                    WHERE model_version = ?
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """
                params = (model_version, limit)
            else:
                query = """
                    SELECT * FROM model_performance 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """
                params = (limit,)
            
            df = pd.read_sql_query(query, conn, params=params)
        
        return df
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess the quality of the dataset."""
        if df.empty:
            return {'status': 'empty_dataset'}
        
        quality_metrics = {
            'completeness': 1.0 - (df.isnull().sum().sum() / (len(df) * len(df.columns))),
            'time_gaps': self._detect_time_gaps(df),
            'outlier_ratio': self._detect_outliers(df),
            'duplicate_ratio': df.duplicated().sum() / len(df)
        }
        
        # Overall quality score
        quality_score = (
            0.4 * quality_metrics['completeness'] +
            0.3 * (1.0 - min(1.0, quality_metrics['outlier_ratio'])) +
            0.3 * (1.0 - quality_metrics['duplicate_ratio'])
        )
        
        quality_metrics['overall_score'] = quality_score
        quality_metrics['status'] = 'good' if quality_score > 0.8 else 'needs_attention'
        
        return quality_metrics
    
    def _detect_time_gaps(self, df: pd.DataFrame) -> float:
        """Detect gaps in time series data."""
        if 'timestamp' not in df.columns or len(df) < 2:
            return 0.0
        
        time_diffs = np.diff(df['timestamp'].sort_values())
        median_diff = np.median(time_diffs)
        
        # Count significant gaps (> 3x median interval)
        large_gaps = np.sum(time_diffs > 3 * median_diff)
        return large_gaps / len(time_diffs)
    
    def _detect_outliers(self, df: pd.DataFrame) -> float:
        """Detect outliers using IQR method."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        total_outliers = 0
        total_values = 0
        
        for col in numeric_columns:
            if col == 'timestamp':
                continue
                
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            total_outliers += len(outliers)
            total_values += len(df)
        
        return total_outliers / max(1, total_values)
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get overall database statistics."""
        with sqlite3.connect(self.db_path) as conn:
            # Process data stats
            process_stats = conn.execute("""
                SELECT 
                    COUNT(*) as total_records,
                    MIN(timestamp) as earliest_record,
                    MAX(timestamp) as latest_record
                FROM process_data
            """).fetchone()
            
            # Model performance stats
            model_stats = conn.execute("""
                SELECT 
                    COUNT(*) as total_training_runs,
                    COUNT(DISTINCT model_version) as unique_models,
                    AVG(validation_loss) as avg_validation_loss
                FROM model_performance
            """).fetchone()
        
        return {
            'process_data': {
                'total_records': process_stats[0],
                'time_span_hours': (process_stats[2] - process_stats[1]) / 3600 if process_stats[1] else 0,
                'earliest_record': datetime.fromtimestamp(process_stats[1]) if process_stats[1] else None,
                'latest_record': datetime.fromtimestamp(process_stats[2]) if process_stats[2] else None
            },
            'model_performance': {
                'total_training_runs': model_stats[0],
                'unique_models': model_stats[1],
                'average_validation_loss': model_stats[2]
            }
        }
```

### **7. Module: `learning/online_trainer.py`**
**Purpose:** Handles continuous model training and updates.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
import joblib
import os
from datetime import datetime
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error

from ..common.types import TrainingMetrics
from ..models.probabilistic import ProbabilisticTransformer

class OnlineTrainer:
    """
    Manages continuous model training, validation, and deployment.
    Handles model versioning and performance monitoring.
    """
    
    def __init__(self, 
                 model_registry_path: str, 
                 config: Dict[str, Any]):
        """
        Initialize the online trainer.
        
        Args:
            model_registry_path: Path to store versioned models
            config: Training configuration
        """
        self.model_registry_path = model_registry_path
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create registry directory
        os.makedirs(model_registry_path, exist_ok=True)
        
        # Training history
        self.training_history = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def should_retrain(self, 
                      current_performance: Dict[str, float],
                      threshold_config: Dict[str, float]) -> bool:
        """
        Determine if model retraining is needed based on performance metrics.
        
        Args:
            current_performance: Current model performance metrics
            threshold_config: Performance thresholds for triggering retraining
            
        Returns:
            bool: True if retraining is recommended
        """
        # Check validation loss degradation
        validation_loss = current_performance.get('validation_loss', float('inf'))
        loss_threshold = threshold_config.get('max_validation_loss', 0.1)
        
        if validation_loss > loss_threshold:
            self.logger.info(f"Retraining triggered: validation_loss {validation_loss:.4f} > {loss_threshold}")
            return True
        
        # Check prediction accuracy
        prediction_accuracy = current_performance.get('prediction_accuracy', 0.0)
        accuracy_threshold = threshold_config.get('min_prediction_accuracy', 0.85)
        
        if prediction_accuracy < accuracy_threshold:
            self.logger.info(f"Retraining triggered: prediction_accuracy {prediction_accuracy:.4f} < {accuracy_threshold}")
            return True
        
        # Check time since last training
        last_training_time = current_performance.get('last_training_timestamp', 0)
        current_time = datetime.now().timestamp()
        max_training_interval = threshold_config.get('max_training_interval_hours', 24) * 3600
        
        if (current_time - last_training_time) > max_training_interval:
            self.logger.info(f"Retraining triggered: training interval exceeded")
            return True
        
        return False
    
    def run_training_job(self, 
                        training_data: pd.DataFrame, 
                        validation_data: pd.DataFrame,
                        current_model: Optional[ProbabilisticTransformer] = None) -> Tuple[ProbabilisticTransformer, TrainingMetrics]:
        """
        Execute a complete training and validation run.
        
        Args:
            training_data: Training dataset
            validation_data: Validation dataset  
            current_model: Existing model to fine-tune (optional)
            
        Returns:
            Tuple[ProbabilisticTransformer, TrainingMetrics]: (new_model, metrics)
        """
        start_time = datetime.now()
        self.logger.info(f"Starting training job with {len(training_data)} training samples")
        
        # Prepare datasets
        train_loader, val_loader = self._prepare_dataloaders(training_data, validation_data)
        
        # Initialize or load model
        if current_model is None:
            model = self._initialize_new_model()
        else:
            model = self._prepare_existing_model(current_model)
        
        model = model.to(self.device)
        
        # Setup training components
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.get('learning_rate', 0.001),
            weight_decay=self.config.get('weight_decay', 1e-5)
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=self.config.get('lr_patience', 5),
            verbose=True
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = self.config.get('early_stopping_patience', 10)
        
        for epoch in range(self.config.get('max_epochs', 100)):
            # Training phase
            train_loss = self._train_epoch(model, train_loader, optimizer, criterion)
            
            # Validation phase
            val_loss, val_metrics = self._validate_epoch(model, val_loader, criterion)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= max_patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break
                
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        # Restore best model
        model.load_state_dict(best_model_state)
        
        # Final validation
        final_val_loss, final_metrics = self._validate_epoch(model, val_loader, criterion)
        
        # Create training metrics
        end_time = datetime.now()
        training_duration = (end_time - start_time).total_seconds()
        
        model_version = f"v{int(datetime.now().timestamp())}"
        
        training_metrics = TrainingMetrics(
            model_version=model_version,
            validation_loss=final_val_loss,
            training_duration_seconds=training_duration,
            dataset_size=len(training_data),
            hyperparameters=self.config.copy()
        )
        
        # Save model
        self._save_model(model, model_version, training_metrics)
        
        # Update training history
        self.training_history.append({
            'timestamp': end_time.timestamp(),
            'metrics': training_metrics,
            'final_metrics': final_metrics
        })
        
        self.logger.info(f"Training completed: {model_version}, val_loss={final_val_loss:.4f}")
        
        return model, training_metrics
    
    def _prepare_dataloaders(self, 
                           training_data: pd.DataFrame, 
                           validation_data: pd.DataFrame) -> Tuple[DataLoader, DataLoader]:
        """Prepare PyTorch dataloaders from pandas DataFrames."""
        
        # Define feature columns
        cma_columns = self.config['cma_names']
        cpp_columns = self.config['cpp_names_and_soft_sensors']
        
        # Convert to sequences
        train_sequences = self._create_sequences(training_data, cma_columns, cpp_columns)
        val_sequences = self._create_sequences(validation_data, cma_columns, cpp_columns)
        
        # Create datasets
        train_dataset = TensorDataset(*train_sequences)
        val_dataset = TensorDataset(*val_sequences)
        
        # Create dataloaders
        batch_size = self.config.get('batch_size', 32)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=self.config.get('num_workers', 2)
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=self.config.get('num_workers', 2)
        )
        
        return train_loader, val_loader
    
    def _create_sequences(self, 
                         data: pd.DataFrame, 
                         cma_columns: List[str], 
                         cpp_columns: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert DataFrame to sequence tensors."""
        lookback = self.config['lookback']
        horizon = self.config['horizon']
        
        # Extract data arrays
        cma_data = data[cma_columns].values
        cpp_data = data[cpp_columns].values
        
        # Create sequences
        past_cmas_list = []
        past_cpps_list = []
        future_cpps_list = []
        future_cmas_list = []
        
        for i in range(len(data) - lookback - horizon + 1):
            # Historical data
            past_cmas = cma_data[i:i+lookback]
            past_cpps = cpp_data[i:i+lookback]
            
            # Future data
            future_cpps = cpp_data[i+lookback:i+lookback+horizon]
            future_cmas = cma_data[i+lookback:i+lookback+horizon]
            
            past_cmas_list.append(past_cmas)
            past_cpps_list.append(past_cpps)
            future_cpps_list.append(future_cpps)
            future_cmas_list.append(future_cmas)
        
        # Convert to tensors
        past_cmas_tensor = torch.tensor(np.array(past_cmas_list), dtype=torch.float32)
        past_cpps_tensor = torch.tensor(np.array(past_cpps_list), dtype=torch.float32)
        future_cpps_tensor = torch.tensor(np.array(future_cpps_list), dtype=torch.float32)
        future_cmas_tensor = torch.tensor(np.array(future_cmas_list), dtype=torch.float32)
        
        return past_cmas_tensor, past_cpps_tensor, future_cpps_tensor, future_cmas_tensor
    
    def _initialize_new_model(self) -> ProbabilisticTransformer:
        """Initialize a new model with configured hyperparameters."""
        model_config = self.config.get('model_hyperparameters', {})
        
        model = ProbabilisticTransformer(
            cma_features=len(self.config['cma_names']),
            cpp_features=len(self.config['cpp_names_and_soft_sensors']),
            d_model=model_config.get('d_model', 128),
            nhead=model_config.get('nhead', 8),
            num_encoder_layers=model_config.get('num_encoder_layers', 3),
            num_decoder_layers=model_config.get('num_decoder_layers', 2),
            dim_feedforward=model_config.get('dim_feedforward', 512),
            dropout=model_config.get('dropout', 0.1)
        )
        
        return model
    
    def _prepare_existing_model(self, model: ProbabilisticTransformer) -> ProbabilisticTransformer:
        """Prepare existing model for fine-tuning."""
        # Optionally freeze some layers for transfer learning
        if self.config.get('freeze_encoder', False):
            for param in model.transformer.encoder.parameters():
                param.requires_grad = False
        
        return model
    
    def _train_epoch(self, 
                    model: ProbabilisticTransformer, 
                    dataloader: DataLoader, 
                    optimizer: optim.Optimizer, 
                    criterion: nn.Module) -> float:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for past_cmas, past_cpps, future_cpps, target_cmas in dataloader:
            # Move to device
            past_cmas = past_cmas.to(self.device)
            past_cpps = past_cpps.to(self.device) 
            future_cpps = future_cpps.to(self.device)
            target_cmas = target_cmas.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(past_cmas, past_cpps, future_cpps)
            loss = criterion(predictions, target_cmas)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def _validate_epoch(self, 
                       model: ProbabilisticTransformer, 
                       dataloader: DataLoader, 
                       criterion: nn.Module) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch."""
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for past_cmas, past_cpps, future_cpps, target_cmas in dataloader:
                # Move to device
                past_cmas = past_cmas.to(self.device)
                past_cpps = past_cpps.to(self.device)
                future_cpps = future_cpps.to(self.device)
                target_cmas = target_cmas.to(self.device)
                
                # Forward pass
                predictions = model(past_cmas, past_cpps, future_cpps)
                loss = criterion(predictions, target_cmas)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Store for metrics calculation
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(target_cmas.cpu().numpy())
        
        # Calculate additional metrics
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Flatten for metric calculation
        pred_flat = all_predictions.reshape(-1, all_predictions.shape[-1])
        target_flat = all_targets.reshape(-1, all_targets.shape[-1])
        
        metrics = {
            'mse': mean_squared_error(target_flat, pred_flat),
            'mae': mean_absolute_error(target_flat, pred_flat),
            'prediction_accuracy': self._calculate_prediction_accuracy(pred_flat, target_flat)
        }
        
        avg_loss = total_loss / num_batches
        return avg_loss, metrics
    
    def _calculate_prediction_accuracy(self, predictions: np.ndarray, targets: np.ndarray, tolerance: float = 0.1) -> float:
        """Calculate prediction accuracy within tolerance."""
        relative_errors = np.abs((predictions - targets) / (targets + 1e-8))
        accurate_predictions = relative_errors < tolerance
        return np.mean(accurate_predictions)
    
    def _save_model(self, 
                   model: ProbabilisticTransformer, 
                   model_version: str, 
                   metrics: TrainingMetrics):
        """Save model and metadata to registry."""
        model_dir = os.path.join(self.model_registry_path, model_version)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model state dict
        model_path = os.path.join(model_dir, 'model.pth')
        torch.save(model.state_dict(), model_path)
        
        # Save model architecture info
        model_info = model.get_model_info()
        joblib.dump(model_info, os.path.join(model_dir, 'model_info.pkl'))
        
        # Save training metrics
        joblib.dump(metrics, os.path.join(model_dir, 'training_metrics.pkl'))
        
        # Save training config
        joblib.dump(self.config, os.path.join(model_dir, 'training_config.pkl'))
        
        self.logger.info(f"Model saved: {model_path}")
    
    def load_model(self, model_version: str) -> Tuple[ProbabilisticTransformer, TrainingMetrics]:
        """Load a saved model from registry."""
        model_dir = os.path.join(self.model_registry_path, model_version)
        
        # Load model info and metrics
        model_info = joblib.load(os.path.join(model_dir, 'model_info.pkl'))
        metrics = joblib.load(os.path.join(model_dir, 'training_metrics.pkl'))
        
        # Recreate model
        model = ProbabilisticTransformer(
            cma_features=model_info['cma_features'],
            cpp_features=model_info['cpp_features'],
            d_model=model_info['d_model']
            # Add other hyperparameters as needed
        )
        
        # Load state dict
        model_path = os.path.join(model_dir, 'model.pth')
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        return model, metrics
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get complete training history."""
        return self.training_history.copy()
    
    def get_best_model_version(self) -> Optional[str]:
        """Get the version string of the best performing model."""
        if not self.training_history:
            return None
        
        best_entry = min(self.training_history, 
                        key=lambda x: x['metrics'].validation_loss)
        
        return best_entry['metrics'].model_version
```

### **8. Module: `learning/rl_policy.py`**
**Purpose:** Reinforcement learning policy training (future extension).

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import gym
from gym import spaces
import numpy as np
from typing import Dict, Any, Optional, Tuple
import pickle
import os

class ProcessControlEnv(gym.Env):
    """
    Gym environment wrapper for the pharmaceutical process.
    Enables RL training on the granulation process.
    """
    
    def __init__(self, simulator, config: Dict[str, Any]):
        """
        Initialize the RL environment.
        
        Args:
            simulator: Process simulator (AdvancedPlantSimulator)
            config: Environment configuration
        """
        super(ProcessControlEnv, self).__init__()
        
        self.simulator = simulator
        self.config = config
        
        # Action space: CPP changes
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(config['cpp_names']),), dtype=np.float32
        )
        
        # Observation space: CMAs + CPPs + historical context
        obs_dim = len(config['cma_names']) + len(config['cpp_names']) + config.get('history_length', 5)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # State tracking
        self.current_state = None
        self.target_setpoints = config.get('target_setpoints', {'d50': 380.0, 'lod': 1.8})
        self.step_count = 0
        self.max_steps = config.get('max_steps', 500)
        
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.step_count = 0
        
        # Initialize process at random steady state
        initial_cpps = self._sample_initial_cpps()
        self.current_state = self.simulator.step(initial_cpps)
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        self.step_count += 1
        
        # Convert normalized action to actual CPP changes
        cpp_changes = self._denormalize_action(action)
        
        # Apply action to simulator
        self.current_state = self.simulator.step(cpp_changes)
        
        # Calculate reward
        reward = self._calculate_reward(self.current_state, cpp_changes)
        
        # Check if episode is done
        done = self.step_count >= self.max_steps or self._is_unsafe_state(self.current_state)
        
        # Additional info
        info = {
            'step': self.step_count,
            'cma_values': self.current_state,
            'reward_components': self._get_reward_components(self.current_state, cpp_changes)
        }
        
        return self._get_observation(), reward, done, info
    
    def _sample_initial_cpps(self) -> Dict[str, float]:
        """Sample random initial CPP values within safe ranges."""
        constraints = self.config['process_constraints']
        initial_cpps = {}
        
        for name in self.config['cpp_names']:
            min_val = constraints[name]['min_val']
            max_val = constraints[name]['max_val'] 
            initial_cpps[name] = np.random.uniform(min_val, max_val)
            
        return initial_cpps
    
    def _denormalize_action(self, normalized_action: np.ndarray) -> Dict[str, float]:
        """Convert normalized action [-1, 1] to actual CPP values."""
        constraints = self.config['process_constraints']
        cpp_values = {}
        
        for i, name in enumerate(self.config['cpp_names']):
            max_change = constraints[name]['max_change_per_step']
            change = normalized_action[i] * max_change
            
            # Apply current constraints
            current_val = getattr(self.current_state, name, 100.0)  # Default fallback
            new_val = np.clip(
                current_val + change,
                constraints[name]['min_val'],
                constraints[name]['max_val']
            )
            cpp_values[name] = new_val
            
        return cpp_values
    
    def _get_observation(self) -> np.ndarray:
        """Convert current state to observation vector."""
        obs = []
        
        # Current CMA values
        for name in self.config['cma_names']:
            obs.append(self.current_state.get(name, 0.0))
            
        # Current CPP values  
        for name in self.config['cpp_names']:
            obs.append(self.current_state.get(name, 0.0))
            
        # Add normalized time step
        obs.append(self.step_count / self.max_steps)
        
        return np.array(obs, dtype=np.float32)
    
    def _calculate_reward(self, state: Dict[str, float], actions: Dict[str, float]) -> float:
        """Calculate reward based on process performance."""
        # Tracking error penalty
        tracking_reward = 0.0
        for name, target in self.target_setpoints.items():
            error = abs(state.get(name, 0.0) - target)
            tracking_reward -= error / target  # Normalized error
        
        # Control effort penalty
        control_penalty = 0.0
        for name, value in actions.items():
            control_penalty -= 0.01 * (value ** 2)  # Quadratic penalty
            
        # Safety bonus/penalty
        safety_reward = 0.0 if self._is_safe_state(state) else -10.0
        
        # Combine rewards
        total_reward = tracking_reward + control_penalty + safety_reward
        
        return total_reward
    
    def _get_reward_components(self, state: Dict[str, float], actions: Dict[str, float]) -> Dict[str, float]:
        """Get detailed reward breakdown for analysis."""
        components = {}
        
        # Individual tracking errors
        for name, target in self.target_setpoints.items():
            error = abs(state.get(name, 0.0) - target)
            components[f'{name}_error'] = -error / target
            
        # Control efforts
        for name, value in actions.items():
            components[f'{name}_control'] = -0.01 * (value ** 2)
            
        return components
    
    def _is_safe_state(self, state: Dict[str, float]) -> bool:
        """Check if current state is within safe operating bounds."""
        constraints = self.config['process_constraints']
        
        for name in self.config['cpp_names']:
            value = state.get(name, 0.0)
            if not (constraints[name]['min_val'] <= value <= constraints[name]['max_val']):
                return False
        
        return True
    
    def _is_unsafe_state(self, state: Dict[str, float]) -> bool:
        """Check if state requires immediate termination."""
        return not self._is_safe_state(state)

class RLPolicyTrainer:
    """
    Manages reinforcement learning policy training for process control.
    """
    
    def __init__(self, 
                 simulator,
                 config: Dict[str, Any],
                 policy_registry_path: str):
        """
        Initialize the RL trainer.
        
        Args:
            simulator: Process simulator
            config: Training configuration
            policy_registry_path: Path to store trained policies
        """
        self.simulator = simulator
        self.config = config
        self.policy_registry_path = policy_registry_path
        
        # Create registry directory
        os.makedirs(policy_registry_path, exist_ok=True)
        
        # Initialize environment
        self.env = ProcessControlEnv(simulator, config)
        
    def train_policy(self, 
                    total_timesteps: int = 50000,
                    policy_name: Optional[str] = None) -> PPO:
        """
        Train an RL policy using PPO algorithm.
        
        Args:
            total_timesteps: Number of training timesteps
            policy_name: Name for saving the policy
            
        Returns:
            PPO: Trained policy
        """
        if policy_name is None:
            policy_name = f"ppo_policy_{int(datetime.now().timestamp())}"
            
        # Create vectorized environment
        vec_env = make_vec_env(
            lambda: self.env, 
            n_envs=self.config.get('n_parallel_envs', 4)
        )
        
        # Initialize PPO agent
        model = PPO(
            "MlpPolicy", 
            vec_env,
            learning_rate=self.config.get('rl_learning_rate', 3e-4),
            n_steps=self.config.get('n_steps', 2048),
            batch_size=self.config.get('batch_size', 64),
            n_epochs=self.config.get('n_epochs', 10),
            gamma=self.config.get('gamma', 0.99),
            verbose=1
        )
        
        # Setup evaluation callback
        eval_env = ProcessControlEnv(self.simulator, self.config)
        eval_callback = EvalCallback(
            eval_env, 
            best_model_save_path=os.path.join(self.policy_registry_path, policy_name),
            log_path=os.path.join(self.policy_registry_path, 'logs'),
            eval_freq=self.config.get('eval_freq', 10000)
        )
        
        # Train the model
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback
        )
        
        # Save final model
        model_path = os.path.join(self.policy_registry_path, f"{policy_name}_final")
        model.save(model_path)
        
        # Save training config
        config_path = os.path.join(self.policy_registry_path, f"{policy_name}_config.pkl")
        with open(config_path, 'wb') as f:
            pickle.dump(self.config, f)
        
        return model
    
    def load_policy(self, policy_name: str) -> PPO:
        """Load a trained policy from registry."""
        model_path = os.path.join(self.policy_registry_path, policy_name)
        return PPO.load(model_path)
    
    def evaluate_policy(self, 
                       model: PPO, 
                       n_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate a trained policy's performance.
        
        Args:
            model: Trained PPO model
            n_episodes: Number of evaluation episodes
            
        Returns:
            Dict[str, float]: Performance metrics
        """
        episode_rewards = []
        episode_lengths = []
        tracking_errors = []
        
        for episode in range(n_episodes):
            obs = self.env.reset()
            episode_reward = 0
            episode_length = 0
            episode_tracking_error = 0
            
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                # Calculate tracking error
                cma_values = info['cma_values']
                for name, target in self.env.target_setpoints.items():
                    episode_tracking_error += abs(cma_values.get(name, 0.0) - target)
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            tracking_errors.append(episode_tracking_error / episode_length)
        
        metrics = {
            'mean_episode_reward': np.mean(episode_rewards),
            'std_episode_reward': np.std(episode_rewards),
            'mean_episode_length': np.mean(episode_lengths),
            'mean_tracking_error': np.mean(tracking_errors),
            'policy_performance_score': np.mean(episode_rewards) - np.mean(tracking_errors)
        }
        
        return metrics
```

### **9. Module: `xai/explainer.py`**
**Purpose:** SHAP-based explanations for control decisions.

```python
import shap
import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from ..common.types import StateVector, ControlAction, DecisionExplanation
from ..models.probabilistic import ProbabilisticTransformer

class ShapExplainer:
    """
    Provides SHAP-based explanations for model predictions and control decisions.
    Generates human-interpretable explanations for autonomous control actions.
    """
    
    def __init__(self, 
                 model: ProbabilisticTransformer,
                 training_data_summary: np.ndarray,
                 feature_names: List[str],
                 config: Dict[str, Any]):
        """
        Initialize the SHAP explainer.
        
        Args:
            model: Trained probabilistic transformer model
            training_data_summary: Representative background dataset for SHAP
            feature_names: Names of input features
            config: Explainer configuration
        """
        self.model = model
        self.feature_names = feature_names
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize SHAP explainer
        self.background_data = torch.tensor(training_data_summary, dtype=torch.float32).to(self.device)
        self._initialize_shap_explainer()
        
        # Explanation templates
        self.explanation_templates = self._load_explanation_templates()
        
    def _initialize_shap_explainer(self):
        """Initialize SHAP DeepExplainer for the transformer model."""
        # Create a wrapper function for SHAP
        def model_wrapper(inputs):
            """Wrapper function that SHAP can call."""
            # Assume inputs shape: (batch_size, total_features)
            # Split into past_cmas, past_cpps, future_cpps based on config
            
            lookback = self.config['lookback']
            horizon = self.config['horizon']
            n_cma_features = len(self.config['cma_names'])
            n_cpp_features = len(self.config['cpp_names_and_soft_sensors'])
            
            batch_size = inputs.shape[0]
            
            # Reshape inputs to expected format
            # Expected input format: past_cmas + past_cpps + future_cpps (flattened)
            past_cmas_flat_size = lookback * n_cma_features
            past_cpps_flat_size = lookback * n_cpp_features
            future_cpps_flat_size = horizon * n_cpp_features
            
            past_cmas = inputs[:, :past_cmas_flat_size].reshape(batch_size, lookback, n_cma_features)
            past_cpps = inputs[:, past_cmas_flat_size:past_cmas_flat_size + past_cpps_flat_size].reshape(
                batch_size, lookback, n_cpp_features
            )
            future_cpps = inputs[:, -future_cpps_flat_size:].reshape(batch_size, horizon, n_cpp_features)
            
            # Get model prediction
            with torch.no_grad():
                predictions = self.model(past_cmas, past_cpps, future_cpps)
                
            return predictions.flatten(start_dim=1)  # Flatten for SHAP compatibility
        
        self.model_wrapper = model_wrapper
        
        # Initialize SHAP explainer
        self.explainer = shap.DeepExplainer(model_wrapper, self.background_data)
    
    def explain_prediction(self, 
                          past_cmas: np.ndarray,
                          past_cpps: np.ndarray, 
                          future_cpps: np.ndarray) -> Dict[str, Any]:
        """
        Generate SHAP explanation for a single model prediction.
        
        Args:
            past_cmas: Historical CMA data, shape (lookback, n_cma_features)
            past_cpps: Historical CPP data, shape (lookback, n_cpp_features)  
            future_cpps: Future CPP data, shape (horizon, n_cpp_features)
            
        Returns:
            Dict[str, Any]: SHAP explanation results
        """
        # Flatten inputs for SHAP
        flattened_input = np.concatenate([
            past_cmas.flatten(),
            past_cpps.flatten(), 
            future_cpps.flatten()
        ]).reshape(1, -1)
        
        # Convert to tensor
        input_tensor = torch.tensor(flattened_input, dtype=torch.float32).to(self.device)
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(input_tensor)
        
        # Process SHAP values
        if isinstance(shap_values, list):
            # Multi-output case - take first output for now
            shap_values = shap_values[0]
        
        # Map SHAP values to feature names
        feature_attributions = {}
        for i, name in enumerate(self.feature_names):
            if i < len(shap_values[0]):
                feature_attributions[name] = float(shap_values[0][i])
        
        # Get model prediction for context
        with torch.no_grad():
            past_cmas_tensor = torch.tensor(past_cmas, dtype=torch.float32).unsqueeze(0).to(self.device)
            past_cpps_tensor = torch.tensor(past_cpps, dtype=torch.float32).unsqueeze(0).to(self.device)
            future_cpps_tensor = torch.tensor(future_cpps, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            prediction = self.model(past_cmas_tensor, past_cpps_tensor, future_cpps_tensor)
            prediction_np = prediction.squeeze(0).detach().cpu().numpy()
        
        explanation = {
            'feature_attributions': feature_attributions,
            'prediction': prediction_np,
            'top_positive_features': self._get_top_features(feature_attributions, positive=True),
            'top_negative_features': self._get_top_features(feature_attributions, positive=False),
            'explanation_quality': self._assess_explanation_quality(feature_attributions)
        }
        
        return explanation
    
    def generate_decision_narrative(self, 
                                  history: List[StateVector], 
                                  action: ControlAction,
                                  prediction_explanation: Optional[Dict[str, Any]] = None) -> DecisionExplanation:
        """
        Generate human-readable explanation for a control decision.
        
        Args:
            history: Recent process history
            action: Control action taken
            prediction_explanation: Optional pre-computed SHAP explanation
            
        Returns:
            DecisionExplanation: Complete decision explanation
        """
        # Generate prediction explanation if not provided
        if prediction_explanation is None:
            # Convert history to model input format
            recent_history = history[-self.config['lookback']:]
            past_cmas, past_cpps = self._convert_history_to_arrays(recent_history)
            
            # Create dummy future CPPs for explanation (using current action)
            future_cpps = self._create_future_cpps_from_action(action)
            
            prediction_explanation = self.explain_prediction(past_cmas, past_cpps, future_cpps)
        
        # Generate narrative explanation
        narrative = self._create_narrative_explanation(
            history, action, prediction_explanation
        )
        
        # Calculate confidence factors
        confidence_factors = self._analyze_confidence_factors(
            prediction_explanation, action
        )
        
        decision_explanation = DecisionExplanation(
            decision_id=action.action_id,
            control_action=action,
            narrative=narrative,
            feature_attributions=prediction_explanation['feature_attributions'],
            confidence_factors=confidence_factors,
            alternatives_considered=self._count_alternatives_from_shap(prediction_explanation)
        )
        
        return decision_explanation
    
    def _get_top_features(self, 
                         attributions: Dict[str, float], 
                         positive: bool = True, 
                         n_top: int = 5) -> List[Tuple[str, float]]:
        """Get top contributing features from SHAP attributions."""
        sorted_features = sorted(
            attributions.items(), 
            key=lambda x: x[1] if positive else -x[1], 
            reverse=True
        )
        
        if positive:
            return [(name, value) for name, value in sorted_features[:n_top] if value > 0]
        else:
            return [(name, abs(value)) for name, value in sorted_features[:n_top] if value < 0]
    
    def _assess_explanation_quality(self, attributions: Dict[str, float]) -> Dict[str, float]:
        """Assess the quality and reliability of the explanation."""
        values = list(attributions.values())
        
        quality_metrics = {
            'attribution_magnitude': np.sum(np.abs(values)),
            'attribution_concentration': np.std(values) / (np.mean(np.abs(values)) + 1e-8),
            'n_significant_features': sum(1 for v in values if abs(v) > 0.01),
            'explanation_clarity': min(1.0, np.max(np.abs(values)) / (np.mean(np.abs(values)) + 1e-8))
        }
        
        return quality_metrics
    
    def _convert_history_to_arrays(self, history: List[StateVector]) -> Tuple[np.ndarray, np.ndarray]:
        """Convert StateVector history to numpy arrays."""
        cma_data = []
        cpp_data = []
        
        for state in history:
            # Extract CMAs
            cma_values = [state.cmas[name] for name in self.config['cma_names']]
            cma_data.append(cma_values)
            
            # Extract CPPs (with soft sensors)
            cpp_values = []
            for name in self.config['cpp_names_and_soft_sensors']:
                if name in state.cpps:
                    cpp_values.append(state.cpps[name])
                elif name == 'specific_energy':
                    se = (state.cpps['spray_rate'] * state.cpps['carousel_speed']) / 1000.0
                    cpp_values.append(se)
                elif name == 'froude_number_proxy':
                    fn = (state.cpps['carousel_speed']**2) / 9.81
                    cpp_values.append(fn)
                else:
                    cpp_values.append(0.0)
            cpp_data.append(cpp_values)
        
        return np.array(cma_data), np.array(cpp_data)
    
    def _create_future_cpps_from_action(self, action: ControlAction) -> np.ndarray:
        """Create future CPP sequence from control action."""
        horizon = self.config['horizon']
        n_cpp_features = len(self.config['cpp_names_and_soft_sensors'])
        
        # Create sequence with constant action
        future_cpps = np.zeros((horizon, n_cpp_features))
        
        # Fill in basic CPPs
        for i, name in enumerate(self.config['cpp_names']):
            if name in action.cpp_setpoints:
                future_cpps[:, i] = action.cpp_setpoints[name]
        
        # Calculate soft sensors
        if 'spray_rate' in action.cpp_setpoints and 'carousel_speed' in action.cpp_setpoints:
            spray_rate = action.cpp_setpoints['spray_rate']
            carousel_speed = action.cpp_setpoints['carousel_speed']
            
            specific_energy = (spray_rate * carousel_speed) / 1000.0
            froude_number = (carousel_speed**2) / 9.81
            
            # Find indices for soft sensors
            for i, name in enumerate(self.config['cpp_names_and_soft_sensors']):
                if name == 'specific_energy':
                    future_cpps[:, i] = specific_energy
                elif name == 'froude_number_proxy':
                    future_cpps[:, i] = froude_number
        
        return future_cpps
    
    def _create_narrative_explanation(self, 
                                    history: List[StateVector], 
                                    action: ControlAction,
                                    explanation: Dict[str, Any]) -> str:
        """Create human-readable narrative explanation."""
        # Get current process state
        current_state = history[-1]
        
        # Identify primary control objective
        primary_objective = self._identify_primary_objective(current_state, action)
        
        # Get top influencing factors
        top_positive = explanation['top_positive_features'][:3]
        top_negative = explanation['top_negative_features'][:3]
        
        # Build narrative
        narrative_parts = []
        
        # Opening statement
        narrative_parts.append(f"Control action taken at {datetime.fromtimestamp(action.timestamp).strftime('%H:%M:%S')}:")
        narrative_parts.append(f"Primary objective: {primary_objective}")
        
        # Control actions
        actions_text = []
        for cpp_name, value in action.cpp_setpoints.items():
            current_val = current_state.cpps.get(cpp_name, 0.0)
            change = value - current_val
            direction = "increase" if change > 0 else "decrease" if change < 0 else "maintain"
            actions_text.append(f"{direction} {cpp_name} to {value:.1f}")
        
        narrative_parts.append(f"Actions: {', '.join(actions_text)}")
        
        # Key reasoning
        if top_positive:
            positive_factors = [f"{name} (impact: {value:.3f})" for name, value in top_positive]
            narrative_parts.append(f"Key supporting factors: {', '.join(positive_factors)}")
        
        if top_negative:
            negative_factors = [f"{name} (concern: {value:.3f})" for name, value in top_negative]
            narrative_parts.append(f"Key constraints considered: {', '.join(negative_factors)}")
        
        # Confidence statement
        confidence_pct = int(action.confidence * 100)
        narrative_parts.append(f"Decision confidence: {confidence_pct}%")
        
        return " | ".join(narrative_parts)
    
    def _identify_primary_objective(self, 
                                  current_state: StateVector, 
                                  action: ControlAction) -> str:
        """Identify the primary control objective based on state and action."""
        # This would be more sophisticated in practice
        # For now, identify based on largest action change
        
        max_change = 0
        primary_cpp = ""
        
        for cpp_name, new_value in action.cpp_setpoints.items():
            current_val = current_state.cpps.get(cpp_name, 0.0)
            change = abs(new_value - current_val)
            
            if change > max_change:
                max_change = change
                primary_cpp = cpp_name
        
        # Map CPP to likely objective
        objective_map = {
            'spray_rate': 'particle size control',
            'air_flow': 'moisture content adjustment', 
            'carousel_speed': 'residence time optimization'
        }
        
        return objective_map.get(primary_cpp, 'process optimization')
    
    def _analyze_confidence_factors(self, 
                                  explanation: Dict[str, Any], 
                                  action: ControlAction) -> Dict[str, float]:
        """Analyze factors contributing to decision confidence."""
        attribution_magnitude = explanation['explanation_quality']['attribution_magnitude']
        explanation_clarity = explanation['explanation_quality']['explanation_clarity']
        
        confidence_factors = {
            'model_certainty': min(1.0, attribution_magnitude / 10.0),  # Normalize
            'explanation_clarity': explanation_clarity,
            'feature_consensus': len(explanation['top_positive_features']) / 10.0,
            'action_magnitude': min(1.0, sum(abs(v) for v in action.cpp_setpoints.values()) / 100.0)
        }
        
        return confidence_factors
    
    def _count_alternatives_from_shap(self, explanation: Dict[str, Any]) -> int:
        """Estimate number of alternatives considered based on SHAP analysis."""
        # This is a simplified heuristic
        significant_features = explanation['explanation_quality']['n_significant_features']
        return max(3, significant_features * 2)  # Rough estimate
    
    def _load_explanation_templates(self) -> Dict[str, str]:
        """Load explanation templates for different scenarios."""
        return {
            'tracking_control': "Adjusting {cpp} to {direction} {cma} towards target of {target}",
            'disturbance_rejection': "Countering process disturbance by {action}",
            'optimization': "Optimizing process efficiency through {strategy}",
            'safety_action': "Taking precautionary action to maintain safe operation"
        }
    
    def generate_batch_explanations(self, 
                                  decisions: List[Tuple[List[StateVector], ControlAction]]) -> List[DecisionExplanation]:
        """Generate explanations for multiple decisions efficiently."""
        explanations = []
        
        for history, action in decisions:
            try:
                explanation = self.generate_decision_narrative(history, action)
                explanations.append(explanation)
            except Exception as e:
                # Create fallback explanation
                fallback_explanation = DecisionExplanation(
                    decision_id=action.action_id,
                    control_action=action,
                    narrative=f"Control action executed (explanation generation failed: {str(e)})",
                    feature_attributions={},
                    confidence_factors={'explanation_error': 1.0},
                    alternatives_considered=0
                )
                explanations.append(fallback_explanation)
        
        return explanations
    
    def get_explanation_quality_metrics(self) -> Dict[str, Any]:
        """Get metrics about explanation system performance."""
        return {
            'explainer_type': 'SHAP DeepExplainer',
            'feature_count': len(self.feature_names),
            'background_samples': self.background_data.shape[0],
            'explanation_templates': len(self.explanation_templates)
        }
```

---

## Implementation Roadmap

### Phase 1: Foundation (Months 1-2)
- ✅ Set up V3 directory structure and development environment
- ✅ Create comprehensive design document with detailed APIs
- 🔄 Implement core type definitions (`common/types.py`)
- ⏳ Create enhanced probabilistic transformer model (`models/probabilistic.py`)
- ⏳ Port and enhance V2 control components (`control/estimators.py`, `control/optimizers.py`)
- ⏳ Set up basic testing framework and CI/CD pipeline

### Phase 2: Control System (Months 2-4)
- ⏳ Implement robust MPC controller (`control/mpc.py`)
- ⏳ Create data handler for continuous learning (`learning/data_handler.py`)
- ⏳ Develop online training system (`learning/online_trainer.py`) 
- ⏳ Add safety constraints and monitoring
- ⏳ Create hardware abstraction layer for real systems

### Phase 3: Explainability (Months 3-5)
- ⏳ Implement SHAP-based explainer (`xai/explainer.py`)
- ⏳ Create explanation visualization dashboards
- ⏳ Develop natural language explanation generation
- ⏳ Add audit trail and compliance reporting
- ⏳ Implement multi-modal explanation interfaces

### Phase 4: Advanced Learning (Months 4-6)
- ⏳ Develop reinforcement learning framework (`learning/rl_policy.py`)
- ⏳ Implement transfer learning capabilities
- ⏳ Add hyperparameter optimization service
- ⏳ Create A/B testing framework for model improvements
- ⏳ Implement federated learning across multiple plants

### Phase 5: Integration & Services (Months 5-7)
- ⏳ Build microservices architecture (Control Agent, Learning Service, XAI Service)
- ⏳ Implement FastAPI service endpoints
- ⏳ Add Docker containerization and orchestration
- ⏳ Create monitoring and logging infrastructure
- ⏳ Develop web-based user interface

### Phase 6: Validation & Deployment (Months 6-8)
- ⏳ Full system integration testing
- ⏳ Performance benchmarking against V2
- ⏳ Security and compliance validation
- ⏳ User acceptance testing with domain experts
- ⏳ Production deployment preparation and documentation

---

## Success Criteria & Validation

### Technical Performance Targets
- **Control Performance**: 50% reduction in process variability vs V2
- **Response Time**: < 100ms for real-time control decisions
- **Model Accuracy**: > 95% prediction accuracy on unseen process data
- **Learning Speed**: 80% faster adaptation to process changes vs offline retraining
- **System Uptime**: > 99.9% availability in production environments

### Explainability & Trust Metrics
- **Explanation Quality**: > 90% operator satisfaction with decision explanations
- **Decision Transparency**: 100% of control actions accompanied by interpretable explanations
- **Audit Compliance**: Complete regulatory audit trail for all decisions
- **Uncertainty Communication**: Clear confidence intervals for all predictions
- **Human-AI Collaboration**: Seamless integration with operator workflows

### Business Impact Goals
- **Process Efficiency**: 20% improvement in Overall Equipment Effectiveness (OEE)
- **Quality Consistency**: 40% reduction in product quality variations
- **Operational Costs**: 15% reduction in manufacturing costs through optimization
- **Time to Market**: 30% faster process optimization for new products
- **Regulatory Compliance**: Zero compliance violations related to process control

---

## Conclusion

This comprehensive design document provides the complete blueprint for implementing AutoPharm V3 - an autonomous, explainable, and continuously learning pharmaceutical process control framework. 

**Key Deliverables:**
1. **Complete API Specifications**: Every class, method, and data structure fully defined
2. **Implementation-Ready Code Signatures**: Detailed function signatures with type hints and documentation
3. **Modular Architecture**: Clean separation of concerns across control, learning, and explainability
4. **Production Readiness**: Microservices architecture with containerization and monitoring
5. **Comprehensive Testing Strategy**: Unit tests, integration tests, and performance benchmarks

**Next Steps:**
1. Begin Phase 1 implementation starting with `common/types.py` and `models/probabilistic.py`
2. Set up development environment with proper dependency management
3. Create initial test suite to validate API contracts
4. Implement core control components building on V2 foundations
5. Begin parallel development of learning and explainability modules

**Technical Innovation:**
- **Uncertainty-Aware Control**: Every decision includes confidence quantification
- **Online Learning**: Continuous model improvement without service interruption  
- **Explainable Autonomy**: Human-interpretable explanations for all control actions
- **Multi-Modal AI**: Integration of classical control, deep learning, and reinforcement learning

This design serves as the definitive specification for transforming pharmaceutical manufacturing through intelligent, autonomous, and trustworthy process control.

---

*Document Version: 2.0*  
*Status: Implementation Ready*  
*Total API Surface: 9 modules, 47 public methods, 23 data types*  
*Last Updated: 2025-01-15*
- 🔄 Create basic microservice templates with FastAPI
- ⏳ Set up containerization and orchestration
- ⏳ Implement basic CI/CD pipeline

### Phase 2: Control System (Months 2-4)
- ⏳ Port and enhance V2 MPC controller
- ⏳ Implement real-time control service
- ⏳ Add safety constraints and monitoring
- ⏳ Create hardware abstraction layer
- ⏳ Implement basic explainability features

### Phase 3: Learning System (Months 3-5)  
- ⏳ Build online learning pipeline
- ⏳ Implement model evaluation and monitoring
- ⏳ Add transfer learning capabilities
- ⏳ Create hyperparameter optimization service
- ⏳ Implement A/B testing framework

### Phase 4: Explainability (Months 4-6)
- ⏳ Advanced XAI integration with SHAP
- ⏳ Natural language explanation generation
- ⏳ Interactive visualization dashboards
- ⏳ Audit trail and compliance reporting
- ⏳ Multi-modal explanation interfaces

### Phase 5: Reinforcement Learning (Months 5-7)
- ⏳ Safe RL policy development
- ⏳ Multi-objective optimization
- ⏳ Hierarchical control implementation
- ⏳ Policy verification and validation
- ⏳ Multi-agent coordination

### Phase 6: Integration & Testing (Months 6-8)
- ⏳ Full system integration testing
- ⏳ Performance benchmarking
- ⏳ Security and compliance validation
- ⏳ User acceptance testing
- ⏳ Production deployment preparation

---

## Success Metrics

### Technical Performance
- **Control Performance**: 50% reduction in process variability vs V2
- **Response Time**: < 100ms for control decisions
- **Model Accuracy**: > 95% prediction accuracy on process outcomes  
- **Learning Speed**: 80% faster adaptation to process changes
- **Uptime**: > 99.9% system availability

### Explainability & Trust
- **Explanation Quality**: > 90% operator satisfaction with explanations
- **Decision Transparency**: 100% of control actions explained
- **Audit Compliance**: Full regulatory audit trail maintenance
- **Uncertainty Communication**: Clear confidence intervals for all predictions
- **Human-AI Collaboration**: Seamless integration with operator workflows

### Business Impact
- **Process Efficiency**: 20% improvement in overall equipment effectiveness
- **Quality Consistency**: 40% reduction in product quality variations
- **Operational Costs**: 15% reduction in manufacturing costs
- **Time to Market**: 30% faster process optimization for new products
- **Regulatory Compliance**: Zero compliance violations related to process control

---

## Risk Assessment & Mitigation

### Technical Risks
1. **Model Performance Degradation**
   - *Risk*: Online learning may degrade model performance
   - *Mitigation*: Comprehensive validation pipelines and rollback mechanisms

2. **Real-Time Performance Requirements**
   - *Risk*: Complex AI systems may be too slow for real-time control
   - *Mitigation*: Hierarchical architecture with fast tactical control layer

3. **Explainability Complexity**
   - *Risk*: Advanced AI models may be difficult to explain
   - *Mitigation*: Multi-level explanation framework with varying detail levels

### Operational Risks  
1. **System Integration Complexity**
   - *Risk*: Microservices architecture increases integration complexity
   - *Mitigation*: Comprehensive testing and monitoring infrastructure

2. **Operator Acceptance**
   - *Risk*: Operators may not trust autonomous AI control
   - *Mitigation*: Gradual deployment with extensive explanation and training

### Regulatory Risks
1. **Compliance Requirements**
   - *Risk*: AI systems may not meet regulatory standards
   - *Mitigation*: Built-in audit trails and explainability features

---

## Conclusion

AutoPharm V3 represents the next generation of intelligent manufacturing control systems. By combining autonomous decision-making, continuous learning, and explainable AI, it creates a trustworthy and adaptive framework that can revolutionize pharmaceutical manufacturing.

The system builds upon the solid foundation of V1 (prototype) and V2 (industrial-grade) to create a truly autonomous platform that not only optimizes processes but also explains its reasoning, adapts to changing conditions, and maintains the highest standards of safety and regulatory compliance.

**Next Steps**: Begin implementation of Phase 1 components, starting with the shared core libraries and basic service templates.

---

*Document Version: 1.0*  
*Last Updated: 2025-01-15*  
*Authors: PharmaControl Development Team*