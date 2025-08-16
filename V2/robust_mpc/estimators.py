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
    Enhanced Kalman Filter with bias state augmentation to handle model steady-state offsets.
    
    Mathematical Model:
    - Augmented State: [x_physical, x_bias]
    - State Evolution: x_physical[k+1] = A*x_physical[k] + B*u[k] + x_bias[k] + w1[k]
    - Bias Evolution:  x_bias[k+1] = x_bias[k] + w2[k]  (random walk)
    - Observation:     y[k] = x_physical[k] + v[k]  (bias is in model, not measurement)
    
    The bias states evolve as random walks and are added to the physical state predictions,
    allowing the filter to automatically compensate for systematic model prediction errors.
    This corrects steady-state offsets that arise from linear model approximation of nonlinear dynamics.
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
        
        # Build augmented state-space model
        # Augmented state: [x_physical, x_bias]
        # Model: [x_physical[k+1]] = [[A, I]] * [x_physical[k]] + [B] * u[k] + w[k]
        #        [x_bias[k+1]   ]   [[0, I]]   [x_bias[k]   ]   [0]
        #        y[k] = [I, 0] * [x_physical[k], x_bias[k]]' + v[k]
        
        # Augmented transition matrix A_aug
        A_aug = np.zeros((self.n_augmented_states, self.n_augmented_states))
        A_aug[:self.n_physical_states, :self.n_physical_states] = transition_matrix  # A for physical states
        A_aug[:self.n_physical_states, self.n_physical_states:] = np.eye(self.n_physical_states)  # Bias affects physical states
        A_aug[self.n_physical_states:, self.n_physical_states:] = np.eye(self.n_physical_states)  # Bias evolves as random walk
        
        # Augmented control matrix B_aug
        B_aug = np.zeros((self.n_augmented_states, self.n_controls))
        B_aug[:self.n_physical_states, :] = control_matrix  # B for physical states
        # Bias states are not directly affected by controls
        
        # Augmented observation matrix C_aug
        # We observe only the physical states (bias is in the model, not measurement)
        C_aug = np.zeros((self.n_physical_states, self.n_augmented_states))
        C_aug[:, :self.n_physical_states] = np.eye(self.n_physical_states)  # Observe physical states only
        # Bias states are not directly observed
        
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