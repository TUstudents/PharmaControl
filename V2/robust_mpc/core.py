"""
Robust MPC Core Module

This module contains the main controller classes that orchestrate all components
(state estimation, probabilistic modeling, advanced optimization) into a
cohesive, industrial-grade control system.

Key Classes:
- RobustMPCController: Main V2 controller with all robustness features
- AdaptiveMPCController: Self-tuning and learning capabilities (future)
- EconomicMPCController: Economic optimization integration (future)
- ConstraintMPCController: Formal constraint satisfaction guarantees (future)

Dependencies:
- numpy: Numerical computations
- robust_mpc.estimators: State estimation components
- robust_mpc.models: Probabilistic prediction models
- robust_mpc.optimizers: Advanced optimization algorithms
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
import warnings
from abc import ABC, abstractmethod

# Internal imports - all components now available as of V2-4
from .estimators import KalmanStateEstimator
from .models import ProbabilisticTransformer
from .optimizers import GeneticOptimizer

# Additional imports for the full implementation
import torch
from sklearn.preprocessing import StandardScaler
import warnings


class BaseMPCController(ABC):
    """Abstract base class for all MPC controllers."""
    
    @abstractmethod
    def suggest_action(self, measurement: np.ndarray, setpoint: np.ndarray) -> np.ndarray:
        """Main control loop - returns optimal action."""
        pass


class RobustMPCController(BaseMPCController):
    """
    Industrial-grade Model Predictive Controller with uncertainty awareness,
    offset-free control, and advanced optimization capabilities.
    
    This controller represents the culmination of the V2 series, integrating:
    - Kalman filtering for state estimation
    - Probabilistic models for uncertainty-aware prediction
    - Genetic algorithms for intelligent optimization
    - Integral action for offset-free control
    - Risk-aware cost functions
    
    Implemented in Notebook V2-4.
    """
    
    def __init__(self, 
                 model: ProbabilisticTransformer,
                 estimator: KalmanStateEstimator,
                 optimizer_class,  # GeneticOptimizer class (not instance)
                 config: Dict[str, Any],
                 scalers: Optional[Dict[str, StandardScaler]] = None):
        """
        Initialize the Robust MPC Controller.
        
        Args:
            model: Trained ProbabilisticTransformer model
            estimator: State estimator (Kalman filter)
            optimizer_class: GeneticOptimizer class for instantiation
            config: Configuration dictionary with MPC parameters
            scalers: Optional data scalers for preprocessing
        """
        # Store components
        self.model = model
        self.estimator = estimator  
        self.optimizer_class = optimizer_class
        self.config = config
        self.scalers = scalers or {}
        
        # Set device for PyTorch computations
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Extract key configuration parameters
        self.prediction_horizon = config.get('prediction_horizon', config.get('horizon', 72))
        self.control_horizon = config.get('control_horizon', config.get('horizon', 36))
        self.lookback = config.get('lookback', 36)
        self.integral_gain = config.get('integral_gain', 0.1)
        self.control_effort_weight = config.get('control_effort_weight', 0.05)
        self.risk_aversion_beta = config.get('risk_aversion_beta', config.get('risk_beta', 1.5))
        
        # Problem dimensions
        self.n_cmas = len(config.get('cma_names', ['d50', 'lod']))
        self.n_cpps = len(config.get('cpp_names', ['spray_rate', 'air_flow', 'carousel_speed']))
        self.n_cpp_full = len(config.get('cpp_full_names', 
            ['spray_rate', 'air_flow', 'carousel_speed', 'specific_energy', 'froude_number_proxy']))
        
        # Initialize internal state for integral action
        self.disturbance_estimate = np.zeros(self.n_cmas)
        self.setpoint_history = []
        self.error_history = []
        
        # Data buffers for model input preparation
        self.cma_history = []
        self.cpp_history = []
        
        # Performance tracking
        self.control_history = []
        self.performance_metrics = {
            'total_control_actions': 0,
            'average_control_effort': 0.0,
            'tracking_errors': [],
            'constraint_violations': 0,
            'optimization_times': [],
            'prediction_uncertainties': []
        }
        
        print(f"✅ RobustMPCController initialized successfully!")
        print(f"   - Model: {type(self.model).__name__}")
        print(f"   - Estimator: {type(self.estimator).__name__}")
        print(f"   - Optimizer: {self.optimizer_class.__name__}")
        print(f"   - Horizon: {self.prediction_horizon}, Risk β: {self.risk_aversion_beta}")
    
    def suggest_action(self, 
                      measurement: np.ndarray, 
                      control_input: np.ndarray,
                      setpoint: np.ndarray) -> np.ndarray:
        """
        Main control loop implementing the complete V2 MPC algorithm.
        
        Algorithm:
        1. Filter noisy measurement through Kalman estimator
        2. Update disturbance estimate for integral action
        3. Define risk-aware fitness function using probabilistic model
        4. Optimize control sequence using genetic algorithm
        5. Return first step of optimal plan (receding horizon)
        
        Args:
            measurement: Noisy sensor measurement (n_cmas,)
            control_input: Current control input (n_cpps,)
            setpoint: Desired target values (n_cmas,)
            
        Returns:
            np.ndarray: Optimal control action (n_cpps,)
        """
        import time
        start_time = time.time()
        
        # 1. Filter noisy measurement through Kalman estimator
        filtered_state = self.estimator.estimate(measurement, control_input)
        
        # 2. Update disturbance estimate for integral action
        self.update_disturbance_estimate(setpoint - filtered_state)
        
        # 3. Update data buffers with current measurements
        self._update_data_buffers(filtered_state, control_input)
        
        # 4. Define risk-aware fitness function
        fitness_function = self.define_fitness_function(filtered_state, setpoint)
        
        # 5. Set up optimization bounds and parameters
        bounds = self._get_optimization_bounds()
        
        # 6. Instantiate and run the genetic optimizer
        optimizer = self.optimizer_class(
            population_size=self.config.get('population_size', 50),
            generations=self.config.get('generations', 20),
            mutation_rate=self.config.get('mutation_rate', 0.1),
            crossover_rate=self.config.get('crossover_rate', 0.7)
        )
        
        # Optimize the control sequence
        optimal_solution = optimizer.optimize(fitness_function, bounds)
        
        # 7. Extract first step of optimal plan (receding horizon principle)
        optimal_action = optimal_solution[:self.n_cpps]
        
        # 8. Update performance tracking
        optimization_time = time.time() - start_time
        self._update_performance_metrics(optimal_action, optimization_time)
        
        return optimal_action
    
    def update_disturbance_estimate(self, tracking_error: np.ndarray):
        """
        Update internal disturbance estimate for integral action.
        
        This provides offset-free control by learning unmeasured disturbances.
        
        Args:
            tracking_error: Difference between setpoint and filtered measurement
        """
        # Integral action with anti-windup
        self.disturbance_estimate += self.integral_gain * tracking_error
        
        # Simple anti-windup: limit disturbance estimate magnitude
        max_disturbance = 2.0  # Maximum reasonable disturbance
        self.disturbance_estimate = np.clip(
            self.disturbance_estimate, -max_disturbance, max_disturbance
        )
        
        # Store error history for analysis
        self.error_history.append(tracking_error.copy())
        if len(self.error_history) > 1000:  # Keep last 1000 errors
            self.error_history.pop(0)
    
    def define_fitness_function(self, 
                               filtered_state: np.ndarray, 
                               setpoint: np.ndarray) -> Callable:
        """
        Create risk-aware fitness function for optimization.
        
        Uses Upper Confidence Bound (UCB) approach: cost based on
        mean prediction + beta * prediction uncertainty.
        
        Args:
            filtered_state: Current state estimate from Kalman filter
            setpoint: Target setpoint
            
        Returns:
            Callable: Fitness function for optimizer
        """
        # Prepare historical data for model input
        past_cmas, past_cpps = self._prepare_model_inputs(filtered_state)
        target_trajectory = np.tile(setpoint, (self.prediction_horizon, 1))
        
        def fitness_function(control_plan_flat: np.ndarray) -> float:
            """
            Evaluate the cost of a candidate control plan.
            
            Args:
                control_plan_flat: Flattened control sequence (horizon * n_cpps,)
                
            Returns:
                float: Total cost (lower is better)
            """
            try:
                # Reshape flat control plan to matrix form
                control_plan = control_plan_flat.reshape(self.prediction_horizon, self.n_cpps)
                
                # Add soft sensors to control plan
                control_plan_with_sensors = self._add_soft_sensors(control_plan)
                
                # Scale inputs for model
                past_cmas_scaled = self._scale_data(past_cmas, 'cma')
                past_cpps_scaled = self._scale_data(past_cpps, 'cpp')
                future_cpps_scaled = self._scale_data(control_plan_with_sensors, 'cpp')
                
                # Convert to tensors
                past_cmas_tensor = torch.tensor(past_cmas_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
                past_cpps_tensor = torch.tensor(past_cpps_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
                future_cpps_tensor = torch.tensor(future_cpps_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # Get probabilistic prediction
                with torch.no_grad():
                    mean_pred, std_pred = self.model.predict_distribution(
                        past_cmas_tensor, past_cpps_tensor, future_cpps_tensor,
                        n_samples=self.config.get('mc_samples', 30)
                    )
                
                # Convert back to numpy
                mean_pred_np = mean_pred.squeeze(0).cpu().numpy()
                std_pred_np = std_pred.squeeze(0).cpu().numpy()
                
                # Apply disturbance correction (integral action)
                corrected_prediction = mean_pred_np + self.disturbance_estimate
                
                # Calculate risk-adjusted prediction (Upper Confidence Bound)
                risk_adjusted_pred = upper_confidence_bound(
                    corrected_prediction, std_pred_np, self.risk_aversion_beta
                )
                
                # Calculate tracking error cost
                tracking_cost = calculate_tracking_error(risk_adjusted_pred, target_trajectory)
                
                # Calculate control effort penalty
                control_effort_cost = calculate_control_effort(control_plan) * self.control_effort_weight
                
                # Total cost
                total_cost = tracking_cost + control_effort_cost
                
                # Store prediction uncertainty for analysis
                avg_uncertainty = np.mean(std_pred_np)
                self.performance_metrics['prediction_uncertainties'].append(avg_uncertainty)
                
                return float(total_cost)
                
            except Exception as e:
                # Return high cost for invalid solutions
                warnings.warn(f"Fitness evaluation failed: {e}")
                return 1e6
        
        return fitness_function
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Return comprehensive performance statistics.
        
        Returns:
            Dict containing control performance metrics
        """
        metrics = self.performance_metrics.copy()
        
        # Calculate derived statistics
        if metrics['tracking_errors']:
            metrics['mean_tracking_error'] = np.mean(metrics['tracking_errors'])
            metrics['std_tracking_error'] = np.std(metrics['tracking_errors'])
        
        if metrics['optimization_times']:
            metrics['mean_optimization_time'] = np.mean(metrics['optimization_times'])
            metrics['max_optimization_time'] = np.max(metrics['optimization_times'])
        
        if metrics['prediction_uncertainties']:
            metrics['mean_prediction_uncertainty'] = np.mean(metrics['prediction_uncertainties'])
        
        # Controller state information
        metrics['current_disturbance_estimate'] = self.disturbance_estimate.copy()
        metrics['disturbance_magnitude'] = np.linalg.norm(self.disturbance_estimate)
        
        return metrics
    
    def reset(self):
        """Reset controller internal state."""
        # Reset integral action
        self.disturbance_estimate = np.zeros(self.n_cmas)
        
        # Clear history buffers
        self.setpoint_history = []
        self.error_history = []
        self.control_history = []
        self.cma_history = []
        self.cpp_history = []
        
        # Reset performance metrics
        self.performance_metrics = {
            'total_control_actions': 0,
            'average_control_effort': 0.0,
            'tracking_errors': [],
            'constraint_violations': 0,
            'optimization_times': [],
            'prediction_uncertainties': []
        }
        
        # Reset estimator state
        if hasattr(self.estimator, 'reset'):
            self.estimator.reset()
    
    # Helper methods for internal functionality
    def _update_data_buffers(self, filtered_state: np.ndarray, control_input: np.ndarray):
        """Update internal data buffers for model input preparation."""
        self.cma_history.append(filtered_state.copy())
        self.cpp_history.append(control_input.copy())
        
        # Keep only required lookback length
        if len(self.cma_history) > self.lookback:
            self.cma_history.pop(0)
            self.cpp_history.pop(0)
    
    def _prepare_model_inputs(self, current_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare historical data for model input."""
        # If not enough history, pad with current state
        if len(self.cma_history) < self.lookback:
            past_cmas = np.tile(current_state, (self.lookback, 1))
            past_cpps = np.tile([120.0, 500.0, 30.0], (self.lookback, 1))  # Default CPPs
        else:
            past_cmas = np.array(self.cma_history[-self.lookback:])
            past_cpps = np.array(self.cpp_history[-self.lookback:])
        
        # Add soft sensors to CPPs
        past_cpps_with_sensors = self._add_soft_sensors(past_cpps)
        
        return past_cmas, past_cpps_with_sensors
    
    def _add_soft_sensors(self, cpp_data: np.ndarray) -> np.ndarray:
        """Add soft sensor features to CPP data."""
        # Extract basic CPPs
        spray_rate = cpp_data[:, 0]
        air_flow = cpp_data[:, 1] 
        carousel_speed = cpp_data[:, 2]
        
        # Calculate soft sensors (simplified physics-based features)
        specific_energy = spray_rate * 0.1 + air_flow * 0.001  # Simplified energy calculation
        froude_number_proxy = carousel_speed**2 / (carousel_speed + 10)  # Simplified Froude number
        
        # Combine all features
        cpp_with_sensors = np.column_stack([
            spray_rate, air_flow, carousel_speed, specific_energy, froude_number_proxy
        ])
        
        return cpp_with_sensors
    
    def _scale_data(self, data: np.ndarray, data_type: str) -> np.ndarray:
        """Scale data using stored scalers or simple normalization."""
        if self.scalers and data_type in self.scalers:
            return self.scalers[data_type].transform(data)
        else:
            # Simple min-max normalization as fallback
            if data_type == 'cma':
                # Typical ranges for CMAs
                data_min = np.array([50.0, 0.5])  # d50, LOD
                data_max = np.array([200.0, 5.0])
            else:  # cpp
                # Typical ranges for CPPs with soft sensors  
                data_min = np.array([80.0, 400.0, 20.0, 0.0, 0.0])
                data_max = np.array([180.0, 700.0, 40.0, 50.0, 30.0])
            
            return (data - data_min) / (data_max - data_min)
    
    def _get_optimization_bounds(self) -> List[Tuple[float, float]]:
        """Get optimization bounds for the genetic algorithm."""
        bounds = []
        cpp_constraints = self.config.get('cpp_constraints', {})
        cpp_names = self.config.get('cpp_names', ['spray_rate', 'air_flow', 'carousel_speed'])
        
        # Default constraints if not provided
        default_constraints = {
            'spray_rate': {'min_val': 80.0, 'max_val': 180.0},
            'air_flow': {'min_val': 400.0, 'max_val': 700.0}, 
            'carousel_speed': {'min_val': 20.0, 'max_val': 40.0}
        }
        
        for _ in range(self.prediction_horizon):
            for cpp_name in cpp_names:
                constraints = cpp_constraints.get(cpp_name, default_constraints.get(cpp_name, {'min_val': 0, 'max_val': 100}))
                bounds.append((constraints['min_val'], constraints['max_val']))
        
        return bounds
    
    def _update_performance_metrics(self, action: np.ndarray, optimization_time: float):
        """Update internal performance tracking."""
        self.control_history.append(action.copy())
        self.performance_metrics['total_control_actions'] += 1
        self.performance_metrics['optimization_times'].append(optimization_time)
        
        # Calculate control effort
        if len(self.control_history) > 1:
            effort = np.sum((action - self.control_history[-2])**2)
            self.performance_metrics['average_control_effort'] = (
                (self.performance_metrics['average_control_effort'] * (len(self.control_history) - 2) + effort) / 
                (len(self.control_history) - 1)
            )


class AdaptiveMPCController(RobustMPCController):
    """
    Self-adapting MPC controller with online learning capabilities.
    
    Extensions beyond RobustMPCController:
    - Online model recalibration
    - Adaptive parameter tuning
    - Performance degradation detection
    - Automatic recovery strategies
    
    Note: This is a placeholder for future implementation (V2.5+).
    """
    
    def __init__(self):
        raise NotImplementedError(
            "AdaptiveMPCController is planned for future implementation (V2.5+). "
            "Use RobustMPCController for current advanced control needs."
        )


class EconomicMPCController(RobustMPCController):
    """
    MPC controller with integrated economic optimization.
    
    Extensions beyond RobustMPCController:
    - Multi-objective optimization (quality vs. cost vs. throughput)
    - Economic cost functions
    - Supply chain integration
    - Dynamic pricing consideration
    
    Note: This is a placeholder for future implementation (V2.6+).
    """
    
    def __init__(self):
        raise NotImplementedError(
            "EconomicMPCController is planned for future implementation (V2.6+). "
            "Use RobustMPCController for current advanced control needs."
        )


class ConstraintMPCController(RobustMPCController):
    """
    MPC controller with formal constraint satisfaction guarantees.
    
    Extensions beyond RobustMPCController:
    - Tube MPC for constraint satisfaction under uncertainty
    - Robust constraint handling
    - Safety-critical system compliance
    - Formal verification capabilities
    
    Note: This is a placeholder for future implementation (V2.7+).
    """
    
    def __init__(self):
        raise NotImplementedError(
            "ConstraintMPCController is planned for future implementation (V2.7+). "
            "Use RobustMPCController for current advanced control needs."
        )


# Utility functions for MPC
def calculate_control_effort(control_sequence: np.ndarray) -> float:
    """
    Calculate control effort metric (penalizes large/rapid changes).
    
    Args:
        control_sequence: Sequence of control actions
        
    Returns:
        float: Control effort metric
    """
    # Sum of squared differences between consecutive actions
    if len(control_sequence.shape) == 1:
        return np.sum(np.diff(control_sequence)**2)
    else:
        return np.sum(np.diff(control_sequence, axis=0)**2)


def validate_constraints(solution: np.ndarray, 
                        constraints: Dict[str, Dict[str, float]]) -> Tuple[bool, List[str]]:
    """
    Validate that solution satisfies all specified constraints.
    
    Args:
        solution: Control solution to validate
        constraints: Dictionary of constraint specifications
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_violations)
    """
    violations = []
    
    # Check bounds constraints
    for i, var_name in enumerate(constraints.keys()):
        if i >= len(solution):
            break
            
        min_val = constraints[var_name].get('min_val', -np.inf)
        max_val = constraints[var_name].get('max_val', np.inf)
        
        if solution[i] < min_val:
            violations.append(f"{var_name} below minimum ({solution[i]:.2f} < {min_val})")
        elif solution[i] > max_val:
            violations.append(f"{var_name} above maximum ({solution[i]:.2f} > {max_val})")
    
    return len(violations) == 0, violations


def upper_confidence_bound(prediction_mean: np.ndarray, 
                          prediction_std: np.ndarray, 
                          beta: float = 1.0) -> np.ndarray:
    """
    Calculate Upper Confidence Bound for risk-aware optimization.
    
    Args:
        prediction_mean: Mean prediction
        prediction_std: Prediction uncertainty (standard deviation)
        beta: Risk aversion parameter (higher = more conservative)
        
    Returns:
        np.ndarray: Upper confidence bound
    """
    return prediction_mean + beta * prediction_std


def calculate_tracking_error(prediction: np.ndarray, 
                           setpoint: np.ndarray,
                           weights: Optional[np.ndarray] = None) -> float:
    """
    Calculate weighted tracking error between prediction and setpoint.
    
    Args:
        prediction: Predicted trajectory
        setpoint: Desired trajectory  
        weights: Optional weights for different time steps/variables
        
    Returns:
        float: Weighted tracking error
    """
    error = prediction - setpoint
    
    if weights is not None:
        error = error * weights
        
    return np.sum(error**2)