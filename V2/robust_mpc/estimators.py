import numpy as np
from pykalman import KalmanFilter

class KalmanStateEstimator:
    """
    A wrapper around pykalman's KalmanFilter to provide a simple interface
    for state estimation in our control loop.
    """
    def __init__(self, transition_matrix, control_matrix, initial_state_mean, process_noise_std=0.5, measurement_noise_std=10.0):
        n_dim_state, n_dim_ctrl = control_matrix.shape

        self.kf = KalmanFilter(
            n_dim_state=n_dim_state,
            n_dim_obs=n_dim_state, # We observe the full state
            transition_matrices=transition_matrix,
            transition_covariance=np.eye(n_dim_state) * process_noise_std**2, # How much we trust the model
            observation_matrices=np.eye(n_dim_state), # We directly measure the state
            observation_covariance=np.eye(n_dim_state) * measurement_noise_std**2, # How much we trust the measurement
            initial_state_mean=initial_state_mean,
            initial_state_covariance=np.eye(n_dim_state)
        )
        self.control_matrix = control_matrix

        # Initialize filtered state and covariance
        self.filtered_state_mean = self.kf.initial_state_mean
        self.filtered_state_covariance = self.kf.initial_state_covariance

    def estimate(self, measurement, control_input):
        """
        Performs one step of the Kalman Filter's predict-update cycle.

        Args:
            measurement (np.array): The noisy measurement from the sensors.
            control_input (np.array): The control action applied at this step.

        Returns:
            np.array: The new, filtered state estimate.
        """
        # The control input needs to be incorporated into the transition offset
        transition_offset = np.dot(self.control_matrix, control_input)

        # Use the filter's update method, which internally performs both predict and update
        self.filtered_state_mean, self.filtered_state_covariance = self.kf.filter_update(
            filtered_state_mean=self.filtered_state_mean,
            filtered_state_covariance=self.filtered_state_covariance,
            observation=measurement,
            transition_offset=transition_offset
        )

        return self.filtered_state_mean

class BiasAugmentedKalmanStateEstimator:
    """
    Enhanced Kalman Filter with bias state augmentation for PROCESS BIAS correction.
    
    This models systematic errors in the process dynamics themselves, such as missing 
    intercept terms and unmodeled nonlinearities in pharmaceutical granulation processes.
    
    Mathematical Model (PROCESS BIAS):
    - Augmented State: [x_physical, x_bias]
    - State Evolution: x_physical[k+1] = A*x_physical[k] + B*u[k] + x_bias[k] + w1[k]  (bias affects dynamics)
    - Bias Evolution:  x_bias[k+1] = x_bias[k] + w2[k]  (random walk)
    - Observation:     y[k] = x_physical[k] + v[k]  (sensors are unbiased, observe physical state only)
    
    Key Insight: The bias represents systematic model errors (e.g., missing sklearn intercept), 
    NOT sensor calibration errors. This distinguishes it from measurement bias models where
    bias appears in the observation equation instead of state evolution.
    """
    def __init__(self, transition_matrix, control_matrix, initial_state_mean, 
                 process_noise_std=0.5, measurement_noise_std=10.0, bias_process_noise_std=0.1):
        """
        Initialize bias-augmented Kalman filter.
        
        Args:
            transition_matrix: Original A matrix (n_states x n_states)
            control_matrix: Original B matrix (n_states x n_controls)
            initial_state_mean: Initial physical state estimate
            process_noise_std: Process noise for physical states
            measurement_noise_std: Sensor measurement noise
            bias_process_noise_std: Process noise for bias states (smaller = slower adaptation)
        """
        self.n_physical_states = len(initial_state_mean)
        self.n_controls = control_matrix.shape[1]
        self.n_augmented_states = 2 * self.n_physical_states  # physical + bias states
        
        # Store original matrices for reference
        self.original_A = transition_matrix
        self.original_B = control_matrix
        
        # Build augmented state-space model for PROCESS BIAS
        # Augmented state: [x_physical, x_bias]
        # Model: [x_physical[k+1]] = [[A, I]] * [x_physical[k]] + [B] * u[k] + w[k]
        #        [x_bias[k+1]   ]   [[0, I]]   [x_bias[k]   ]   [0]
        #        y[k] = [I, 0] * [x_physical[k], x_bias[k]]' + v[k]
        
        # Augmented transition matrix A_aug
        A_aug = np.zeros((self.n_augmented_states, self.n_augmented_states))
        A_aug[:self.n_physical_states, :self.n_physical_states] = transition_matrix  # A for physical states
        A_aug[:self.n_physical_states, self.n_physical_states:] = np.eye(self.n_physical_states)  # Bias affects physical state evolution
        A_aug[self.n_physical_states:, self.n_physical_states:] = np.eye(self.n_physical_states)  # Bias evolves as random walk
        
        # Augmented control matrix B_aug
        B_aug = np.zeros((self.n_augmented_states, self.n_controls))
        B_aug[:self.n_physical_states, :] = control_matrix  # B for physical states
        # Bias states are not directly affected by controls
        
        # Augmented observation matrix C_aug
        # CORRECTED FOR PROCESS BIAS: We observe only the physical state (bias affects dynamics, not sensors)
        C_aug = np.zeros((self.n_physical_states, self.n_augmented_states))
        C_aug[:, :self.n_physical_states] = np.eye(self.n_physical_states)   # Observe physical state only
        # C_aug[:, self.n_physical_states:] = 0  (bias states not directly observed)
        
        # Augmented process noise covariance Q_aug
        Q_aug = np.zeros((self.n_augmented_states, self.n_augmented_states))
        Q_aug[:self.n_physical_states, :self.n_physical_states] = np.eye(self.n_physical_states) * process_noise_std**2
        Q_aug[self.n_physical_states:, self.n_physical_states:] = np.eye(self.n_physical_states) * bias_process_noise_std**2
        
        # Measurement noise covariance R (unchanged)
        R = np.eye(self.n_physical_states) * measurement_noise_std**2
        
        # Initial augmented state mean and covariance
        initial_augmented_mean = np.zeros(self.n_augmented_states)
        initial_augmented_mean[:self.n_physical_states] = initial_state_mean
        # Initialize bias states to zero (will be learned)
        
        initial_augmented_cov = np.eye(self.n_augmented_states)
        initial_augmented_cov[:self.n_physical_states, :self.n_physical_states] *= 1.0  # Low uncertainty for physical states
        initial_augmented_cov[self.n_physical_states:, self.n_physical_states:] *= 100.0  # High uncertainty for unknown bias
        
        # Create the augmented Kalman filter
        self.kf = KalmanFilter(
            n_dim_state=self.n_augmented_states,
            n_dim_obs=self.n_physical_states,
            transition_matrices=A_aug,
            transition_covariance=Q_aug,
            observation_matrices=C_aug,
            observation_covariance=R,
            initial_state_mean=initial_augmented_mean,
            initial_state_covariance=initial_augmented_cov
        )
        
        self.control_matrix = B_aug
        
        # Initialize filtered state and covariance
        self.filtered_state_mean = self.kf.initial_state_mean
        self.filtered_state_covariance = self.kf.initial_state_covariance
        
    def estimate(self, measurement, control_input):
        """
        Performs one step of the bias-augmented Kalman Filter.
        
        Args:
            measurement (np.array): Noisy measurement of physical states only
            control_input (np.array): Control action applied at this step
            
        Returns:
            tuple: (physical_state_estimate, bias_estimate)
        """
        # The control input needs to be incorporated into the transition offset
        transition_offset = np.dot(self.control_matrix, control_input)
        
        # Update the filter
        self.filtered_state_mean, self.filtered_state_covariance = self.kf.filter_update(
            filtered_state_mean=self.filtered_state_mean,
            filtered_state_covariance=self.filtered_state_covariance,
            observation=measurement,
            transition_offset=transition_offset
        )
        
        # Extract physical states and bias estimates
        physical_states = self.filtered_state_mean[:self.n_physical_states]
        bias_states = self.filtered_state_mean[self.n_physical_states:]
        
        return physical_states, bias_states
    
    def get_bias_corrected_estimate(self, measurement, control_input):
        """
        Get bias-corrected physical state estimate.
        
        Args:
            measurement (np.array): Noisy measurement of physical states
            control_input (np.array): Control action applied
            
        Returns:
            np.array: Bias-corrected physical state estimate
        """
        physical_states, bias_states = self.estimate(measurement, control_input)
        # The physical state estimate is already bias-corrected because:
        # - The model includes bias in state evolution: x[k+1] = A*x[k] + B*u[k] + bias[k]
        # - The Kalman filter learns the bias and incorporates it into state prediction
        # - So physical_state estimate compensates for systematic model errors
        return physical_states
    
    def get_current_bias_estimate(self):
        """
        Get current bias state estimates.
        
        Returns:
            np.array: Current bias estimates for each physical state
        """
        return self.filtered_state_mean[self.n_physical_states:]


class ProcessBiasKalmanEstimator:
    """
    Standard Kalman Filter with a fixed intercept term for PROCESS BIAS correction.
    
    Assumes the system DYNAMICS are biased (e.g., missing intercept from regression).
    Model: x[k+1] = A*x[k] + B*u[k] + intercept + w[k]
           y[k] = C*x[k] + v[k]  (sensors are perfect/unbiased)
    
    This models systematic errors in the process model itself, such as missing 
    intercept terms and unmodeled nonlinearities in pharmaceutical granulation.
    
    Advantages:
    - Simple implementation (no state augmentation)
    - Immediate bias correction if intercept is known accurately
    - Lower computational cost
    - Mathematically consistent with adaptive approach
    
    Disadvantages:
    - No adaptation to changing conditions
    - Performance depends critically on intercept accuracy
    - Cannot handle time-varying process bias
    """
    def __init__(self, transition_matrix, control_matrix, initial_state_mean, intercept_term,
                 process_noise_std=0.5, measurement_noise_std=10.0):
        """
        Initialize intercept-based Kalman filter.
        
        Args:
            transition_matrix: A matrix (n_states x n_states)
            control_matrix: B matrix (n_states x n_controls)
            initial_state_mean: Initial TRUE state estimate (class handles bias correction)
            intercept_term: Fixed intercept from regression (steady-state bias)
            process_noise_std: Process noise for physical states
            measurement_noise_std: Sensor measurement noise
        """
        self.n_states = len(initial_state_mean)
        self.n_controls = control_matrix.shape[1]
        self.intercept = intercept_term
        
        # Store matrices
        self.transition_matrix = transition_matrix
        self.control_matrix = control_matrix
        
        # Initialize filter with the true initial state
        # The class handles bias correction internally through the estimate method
        
        # Create standard Kalman filter (no augmentation)
        self.kf = KalmanFilter(
            n_dim_state=self.n_states,
            n_dim_obs=self.n_states,
            transition_matrices=transition_matrix,
            transition_covariance=np.eye(self.n_states) * process_noise_std**2,
            observation_matrices=np.eye(self.n_states),
            observation_covariance=np.eye(self.n_states) * measurement_noise_std**2,
            initial_state_mean=initial_state_mean,  # Use true initial state directly
            initial_state_covariance=np.eye(self.n_states)
        )
        
        # Initialize filtered state and covariance
        self.filtered_state_mean = self.kf.initial_state_mean
        self.filtered_state_covariance = self.kf.initial_state_covariance
        
    def estimate(self, measurement, control_input):
        """
        Performs one step of the intercept-based Kalman Filter.
        
        CORRECTED IMPLEMENTATION per user feedback: The intercept goes into the transition model.
        Model: x[k+1] = A*x[k] + B*u[k] + intercept + w[k]
        
        This correctly models processes with systematic drift or bias in the dynamics.
        
        Args:
            measurement (np.array): Noisy measurement of physical states  
            control_input (np.array): Control action applied at this step
            
        Returns:
            np.array: Filtered state estimate with intercept correction
        """
        # Calculate control offset
        control_offset = np.dot(self.control_matrix, control_input)
        
        # USER'S CORRECT FIX: Add intercept to transition offset
        # This models: x[k+1] = A*x[k] + B*u[k] + intercept
        transition_offset = control_offset + self.intercept
        
        # Update filter with original measurement and corrected dynamics
        self.filtered_state_mean, self.filtered_state_covariance = self.kf.filter_update(
            filtered_state_mean=self.filtered_state_mean,
            filtered_state_covariance=self.filtered_state_covariance,
            observation=measurement,  # Use original measurement
            transition_offset=transition_offset  # Include intercept in dynamics
        )
        
        # Return filtered state directly (no additional bias correction needed)
        return self.filtered_state_mean
    
    def get_intercept_value(self):
        """
        Get the fixed intercept value being applied.
        
        Returns:
            np.array: Fixed intercept term
        """
        return self.intercept


class MeasurementBiasKalmanEstimator:
    """
    Standard Kalman Filter with a fixed intercept for MEASUREMENT BIAS correction.
    
    Assumes the SENSORS are biased (e.g., calibration offset) but system dynamics are correct.
    Model: x[k+1] = A*x[k] + B*u[k] + w[k]              (unbiased dynamics)
           y[k] = C*x[k] + intercept + v[k]             (biased measurements)
    
    This corrects for systematic sensor offsets, calibration errors, or measurement drift.
    The filter operates on bias-corrected measurements and estimates the true state.
    
    Advantages:
    - Simple implementation (no state augmentation)
    - Immediate bias correction if measurement offset is known
    - Lower computational cost
    - Stable for measurement bias scenarios
    
    Disadvantages:
    - No adaptation to changing measurement bias
    - Performance depends critically on bias knowledge accuracy
    - Cannot handle time-varying sensor drift
    """
    def __init__(self, transition_matrix, control_matrix, initial_state_mean, intercept_term,
                 process_noise_std=0.5, measurement_noise_std=10.0):
        """
        Initialize measurement-bias-corrected Kalman filter.
        
        Args:
            transition_matrix: A matrix (n_states x n_states)
            control_matrix: B matrix (n_states x n_controls)
            initial_state_mean: Initial TRUE state estimate (before bias)
            intercept_term: Fixed measurement bias/offset
            process_noise_std: Process noise for physical states
            measurement_noise_std: Sensor measurement noise (after bias correction)
        """
        self.n_states = len(initial_state_mean)
        self.control_matrix = control_matrix
        self.intercept = intercept_term  # This is the measurement bias
        
        # Create standard Kalman filter for unbiased dynamics
        self.kf = KalmanFilter(
            n_dim_state=self.n_states,
            n_dim_obs=self.n_states,
            transition_matrices=transition_matrix,
            transition_covariance=np.eye(self.n_states) * process_noise_std**2,
            observation_matrices=np.eye(self.n_states),
            observation_covariance=np.eye(self.n_states) * measurement_noise_std**2,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=np.eye(self.n_states)
        )
        
        # Initialize filtered state and covariance
        self.filtered_state_mean = self.kf.initial_state_mean
        self.filtered_state_covariance = self.kf.initial_state_covariance
        
    def estimate(self, measurement, control_input):
        """
        Performs one step of the measurement-bias-corrected Kalman Filter.
        
        THE FIX: Correct the measurement BEFORE the update step by removing known bias.
        The filter then operates on unbiased measurements and unbiased dynamics.
        
        Args:
            measurement (np.array): Biased measurement from sensors
            control_input (np.array): Control action applied at this step
            
        Returns:
            np.array: Filtered true state estimate (bias-corrected)
        """
        # THE CRITICAL FIX: Correct the measurement before processing
        unbiased_measurement = measurement - self.intercept
        
        # Standard control offset (no bias in dynamics)
        transition_offset = np.dot(self.control_matrix, control_input)
        
        # Update filter with corrected measurement and unbiased dynamics
        self.filtered_state_mean, self.filtered_state_covariance = self.kf.filter_update(
            filtered_state_mean=self.filtered_state_mean,
            filtered_state_covariance=self.filtered_state_covariance,
            observation=unbiased_measurement,  # Use corrected measurement
            transition_offset=transition_offset
        )
        
        # Return the true state estimate (already bias-corrected)
        return self.filtered_state_mean
    
    def get_intercept_value(self):
        """
        Get the fixed measurement bias value being corrected.
        
        Returns:
            np.array: Fixed measurement bias term
        """
        return self.intercept