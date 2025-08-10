"""
State Estimation Module

This module provides state estimation algorithms for filtering noisy sensor
measurements and providing clean, reliable state estimates for control systems.

Key Classes:
- KalmanStateEstimator: Linear Kalman Filter for Gaussian systems
- ExtendedKalmanFilter: Nonlinear state estimation (future implementation)
- UnscentedKalmanFilter: Unscented transform for highly nonlinear systems (future)
- ParticleFilter: Particle-based estimation for non-Gaussian systems (future)

Dependencies:
- numpy: Numerical computations
- pykalman: Kalman filtering algorithms
"""

import numpy as np
from pykalman import KalmanFilter
import warnings
from typing import Optional, Tuple, Union

class KalmanStateEstimator:
    """
    A wrapper around pykalman's KalmanFilter to provide a simple interface
    for state estimation in our control loop.
    
    The Kalman Filter optimally combines model predictions with noisy measurements
    to produce the best possible state estimate under Gaussian assumptions.
    
    Attributes:
        kf (KalmanFilter): The underlying pykalman filter object
        control_matrix (np.ndarray): Matrix B relating control inputs to state changes
        filtered_state_mean (np.ndarray): Current filtered state estimate
        filtered_state_covariance (np.ndarray): Current state uncertainty
    """
    
    def __init__(self, 
                 transition_matrix: np.ndarray, 
                 control_matrix: np.ndarray, 
                 initial_state_mean: np.ndarray,
                 process_noise_std: float = 0.5, 
                 measurement_noise_std: float = 10.0,
                 initial_covariance_scale: float = 1.0):
        """
        Initialize the Kalman State Estimator.
        
        Args:
            transition_matrix (np.ndarray): State transition matrix A (n_states x n_states)
            control_matrix (np.ndarray): Control matrix B (n_states x n_controls)  
            initial_state_mean (np.ndarray): Initial state estimate (n_states,)
            process_noise_std (float): Standard deviation of process noise (model uncertainty)
            measurement_noise_std (float): Standard deviation of measurement noise
            initial_covariance_scale (float): Scale factor for initial state uncertainty
            
        Raises:
            ValueError: If matrix dimensions are incompatible
        """
        # Validate inputs
        n_dim_state, n_dim_ctrl = control_matrix.shape
        if transition_matrix.shape != (n_dim_state, n_dim_state):
            raise ValueError(f"Transition matrix must be {n_dim_state}x{n_dim_state}")
        if len(initial_state_mean) != n_dim_state:
            raise ValueError(f"Initial state must have {n_dim_state} elements")

        # Configure the Kalman Filter
        self.kf = KalmanFilter(
            n_dim_state=n_dim_state,
            n_dim_obs=n_dim_state,  # We observe the full state
            transition_matrices=transition_matrix,
            transition_covariance=np.eye(n_dim_state) * process_noise_std**2,
            observation_matrices=np.eye(n_dim_state),  # Direct state observation
            observation_covariance=np.eye(n_dim_state) * measurement_noise_std**2,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=np.eye(n_dim_state) * initial_covariance_scale
        )
        
        # Store important matrices and parameters
        self.control_matrix = control_matrix.copy()
        self.n_dim_state = n_dim_state
        self.n_dim_ctrl = n_dim_ctrl
        self.process_noise_std = process_noise_std
        self.measurement_noise_std = measurement_noise_std
        
        # Initialize filtered state and covariance
        self.filtered_state_mean = self.kf.initial_state_mean.copy()
        self.filtered_state_covariance = self.kf.initial_state_covariance.copy()
        
        # Statistics tracking
        self.step_count = 0
        self.innovation_history = []
        
    def estimate(self, 
                 measurement: np.ndarray, 
                 control_input: np.ndarray) -> np.ndarray:
        """
        Performs one step of the Kalman Filter's predict-update cycle.
        
        Args:
            measurement (np.ndarray): The noisy measurement from sensors (n_states,)
            control_input (np.ndarray): The control action applied (n_controls,)
            
        Returns:
            np.ndarray: The new, filtered state estimate (n_states,)
            
        Raises:
            ValueError: If input dimensions are incorrect
        """
        # Validate inputs
        if len(measurement) != self.n_dim_state:
            raise ValueError(f"Measurement must have {self.n_dim_state} elements")
        if len(control_input) != self.n_dim_ctrl:
            raise ValueError(f"Control input must have {self.n_dim_ctrl} elements")
            
        # Calculate control effect as transition offset
        transition_offset = np.dot(self.control_matrix, control_input)
        
        # Store innovation (measurement residual) for diagnostics
        predicted_measurement = np.dot(
            self.kf.observation_matrices, 
            np.dot(self.kf.transition_matrices, self.filtered_state_mean) + transition_offset
        )
        innovation = measurement - predicted_measurement
        self.innovation_history.append(np.linalg.norm(innovation))
        
        # Perform the filter update
        self.filtered_state_mean, self.filtered_state_covariance = self.kf.filter_update(
            filtered_state_mean=self.filtered_state_mean,
            filtered_state_covariance=self.filtered_state_covariance,
            observation=measurement,
            transition_offset=transition_offset
        )
        
        self.step_count += 1
        return self.filtered_state_mean.copy()
    
    def predict_next_state(self, control_input: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict the next state without updating with a measurement.
        
        Args:
            control_input (np.ndarray): The control action to apply (n_controls,)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (predicted_state, prediction_covariance)
        """
        transition_offset = np.dot(self.control_matrix, control_input)
        
        # Predict step only
        predicted_mean = (np.dot(self.kf.transition_matrices, self.filtered_state_mean) + 
                         transition_offset)
        predicted_covariance = (np.dot(self.kf.transition_matrices, 
                                     np.dot(self.filtered_state_covariance, 
                                           self.kf.transition_matrices.T)) + 
                               self.kf.transition_covariance)
        
        return predicted_mean, predicted_covariance
    
    def get_uncertainty(self) -> np.ndarray:
        """
        Get the current state uncertainty as standard deviations.
        
        Returns:
            np.ndarray: Standard deviation for each state variable
        """
        return np.sqrt(np.diag(self.filtered_state_covariance))
    
    def get_confidence_interval(self, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get confidence intervals for the current state estimate.
        
        Args:
            confidence (float): Confidence level (e.g., 0.95 for 95% confidence)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (lower_bounds, upper_bounds)
        """
        from scipy import stats
        alpha = 1 - confidence
        z_score = stats.norm.ppf(1 - alpha/2)
        
        std_devs = self.get_uncertainty()
        lower_bounds = self.filtered_state_mean - z_score * std_devs
        upper_bounds = self.filtered_state_mean + z_score * std_devs
        
        return lower_bounds, upper_bounds
    
    def is_filter_healthy(self, max_innovation_threshold: float = 3.0) -> bool:
        """
        Check if the filter is performing well based on innovation statistics.
        
        Args:
            max_innovation_threshold (float): Maximum acceptable innovation magnitude
            
        Returns:
            bool: True if filter appears healthy
        """
        if len(self.innovation_history) < 10:
            return True  # Not enough data to judge
            
        recent_innovations = self.innovation_history[-10:]
        mean_innovation = np.mean(recent_innovations)
        
        return mean_innovation < max_innovation_threshold
    
    def reset(self, 
              new_initial_state: Optional[np.ndarray] = None,
              new_initial_covariance: Optional[np.ndarray] = None):
        """
        Reset the filter to initial conditions.
        
        Args:
            new_initial_state (Optional[np.ndarray]): New initial state estimate
            new_initial_covariance (Optional[np.ndarray]): New initial covariance
        """
        if new_initial_state is not None:
            self.filtered_state_mean = new_initial_state.copy()
        else:
            self.filtered_state_mean = self.kf.initial_state_mean.copy()
            
        if new_initial_covariance is not None:
            self.filtered_state_covariance = new_initial_covariance.copy()
        else:
            self.filtered_state_covariance = self.kf.initial_state_covariance.copy()
            
        self.step_count = 0
        self.innovation_history = []
    
    def get_diagnostics(self) -> dict:
        """
        Get filter performance diagnostics.
        
        Returns:
            dict: Dictionary containing filter statistics and health metrics
        """
        diagnostics = {
            'step_count': self.step_count,
            'current_state_uncertainty': self.get_uncertainty(),
            'filter_healthy': self.is_filter_healthy(),
            'process_noise_std': self.process_noise_std,
            'measurement_noise_std': self.measurement_noise_std
        }
        
        if self.innovation_history:
            diagnostics.update({
                'mean_innovation': np.mean(self.innovation_history),
                'std_innovation': np.std(self.innovation_history),
                'latest_innovation': self.innovation_history[-1] if self.innovation_history else 0
            })
            
        return diagnostics


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for nonlinear systems.
    
    Note: This is a placeholder for future implementation.
    The EKF linearizes the nonlinear dynamics at each time step.
    """
    
    def __init__(self):
        raise NotImplementedError("ExtendedKalmanFilter not yet implemented")


class UnscentedKalmanFilter:
    """
    Unscented Kalman Filter for highly nonlinear systems.
    
    Note: This is a placeholder for future implementation.
    The UKF uses the unscented transform to handle nonlinearity.
    """
    
    def __init__(self):
        raise NotImplementedError("UnscentedKalmanFilter not yet implemented")


class ParticleFilter:
    """
    Particle Filter for non-Gaussian systems.
    
    Note: This is a placeholder for future implementation.
    Uses Monte Carlo methods with particle representations.
    """
    
    def __init__(self):
        raise NotImplementedError("ParticleFilter not yet implemented")