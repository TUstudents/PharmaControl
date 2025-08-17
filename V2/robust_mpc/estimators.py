import numpy as np
from pykalman import KalmanFilter

class KalmanStateEstimator:
    """Standard Kalman Filter for pharmaceutical process state estimation.
    
    This class provides a streamlined interface to pykalman's KalmanFilter implementation,
    specifically configured for continuous granulation process control applications.
    Implements the classic linear Kalman filter for optimal state estimation under
    Gaussian noise assumptions.
    
    The filter assumes a linear state-space model:
        x[k+1] = A*x[k] + B*u[k] + w[k]     (state evolution)
        y[k] = C*x[k] + v[k]                 (observation model)
    
    Where:
        - x[k]: Process state vector (e.g., particle size d50, moisture LOD)
        - u[k]: Control input vector (e.g., spray rate, air flow, carousel speed)
        - y[k]: Measurement vector from sensors
        - w[k], v[k]: Process and measurement noise (assumed Gaussian)
    
    Args:
        transition_matrix (np.ndarray): State transition matrix A of shape (n_states, n_states).
            Defines how the process state evolves over time without control input.
        control_matrix (np.ndarray): Control input matrix B of shape (n_states, n_controls).
            Maps control actions to their effect on state evolution.
        initial_state_mean (np.ndarray): Initial state estimate of shape (n_states,).
            Starting point for the filtering process, typically from process knowledge.
        process_noise_std (float, optional): Standard deviation of process noise.
            Lower values indicate higher trust in the state evolution model. Default: 0.5
        measurement_noise_std (float, optional): Standard deviation of measurement noise.
            Should match typical sensor accuracy (e.g., 10-15 μm for particle size). Default: 10.0
    
    Attributes:
        kf (KalmanFilter): Underlying pykalman filter instance with configured parameters
        control_matrix (np.ndarray): Stored control matrix for transition offset computation
        filtered_state_mean (np.ndarray): Current state estimate after latest measurement
        filtered_state_covariance (np.ndarray): Current state uncertainty covariance matrix
    
    Example:
        >>> # Configure for granulation process (d50, LOD states)
        >>> A = np.array([[0.9, 0.05], [0.0, 0.98]])  # Transition matrix
        >>> B = np.array([[0.3, -0.5], [0.0, 0.01]])  # Control matrix  
        >>> initial_state = np.array([400.0, 1.5])    # d50=400μm, LOD=1.5%
        >>> estimator = KalmanStateEstimator(A, B, initial_state)
        >>> 
        >>> # Process measurement and control input
        >>> measurement = np.array([415.2, 1.48])     # Noisy sensor reading
        >>> control = np.array([120.0, 500.0, 30.0]) # spray, air, speed
        >>> filtered_state = estimator.estimate(measurement, control)
    
    Notes:
        - Assumes full state observability (C = I) for pharmaceutical applications
        - Uses identity covariance matrices scaled by noise standard deviations
        - Suitable for processes where systematic bias is not a concern
        - For processes with model bias, consider BiasAugmentedKalmanStateEstimator
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
        """Perform one step of the Kalman filter predict-update cycle.
        
        Executes the complete Kalman filtering algorithm consisting of:
        1. Prediction step: Projects current state estimate forward using process model
        2. Update step: Incorporates new measurement to refine state estimate
        
        The control input is integrated into the prediction through the transition offset,
        allowing the filter to account for known control actions when predicting the
        next state.
        
        Args:
            measurement (np.ndarray): Noisy sensor measurements of shape (n_states,).
                Typical pharmaceutical measurements include particle size and moisture content.
                Units should match the original training data (e.g., μm for d50, % for LOD).
            control_input (np.ndarray): Applied control actions of shape (n_controls,).
                Critical Process Parameters such as spray rate, air flow, and carousel speed.
                Must be in same units and order as used during model training.
        
        Returns:
            np.ndarray: Updated state estimate of shape (n_states,) after incorporating
                the new measurement. Represents the optimal estimate given all available
                information up to the current time step.
        
        Raises:
            ValueError: If measurement or control_input dimensions don't match filter configuration
            RuntimeError: If numerical issues occur during matrix operations
        
        Notes:
            - Automatically handles the predict-update cycle without explicit user calls
            - Updates internal state (filtered_state_mean, filtered_state_covariance)
            - Control offset computation: offset = B @ u to account for control influence
            - Optimal in minimum mean square error sense under linear Gaussian assumptions
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
    """Adaptive Kalman Filter with bias state augmentation for process bias correction.
    
    This advanced estimator addresses systematic model errors in pharmaceutical processes
    by augmenting the state space to include bias states that are learned online.
    Specifically designed for processes where the underlying dynamics contain unmodeled
    components such as missing intercept terms from regression models or unmodeled
    nonlinearities in granulation kinetics.
    
    Mathematical Framework (Process Bias Model):
        Augmented state vector: x_aug = [x_physical, x_bias]
        
        State evolution:
            x_physical[k+1] = A*x_physical[k] + B*u[k] + x_bias[k] + w_physical[k]
            x_bias[k+1] = x_bias[k] + w_bias[k]
        
        Observation model:
            y[k] = x_physical[k] + v[k]
        
        Augmented matrices:
            A_aug = [[A, I],    C_aug = [I, 0]
                     [0, I]]
    
    Where:
        - x_physical: Observable process states (e.g., particle size, moisture content)  
        - x_bias: Latent bias states representing systematic model errors
        - u[k]: Control inputs (Critical Process Parameters)
        - y[k]: Sensor measurements (assumed unbiased)
        - w_physical, w_bias: Independent Gaussian process noise components
        - v[k]: Measurement noise (Gaussian, zero-mean)
    
    Key Advantages:
        - Online learning of systematic model bias without prior knowledge
        - Maintains optimal filtering performance while correcting for model mismatch
        - Robust to changing process conditions through adaptive bias estimation
        - Mathematically sound augmentation preserves Kalman filter optimality
    
    Applications:
        - Pharmaceutical granulation with missing sklearn regression intercepts
        - Processes with slow-varying disturbances or unmeasured inputs
        - Systems where linear models approximate nonlinear dynamics
        - Any scenario requiring bias-corrected state estimation
    
    Args:
        transition_matrix (np.ndarray): Physical state transition matrix A of shape (n_states, n_states).
            Describes the nominal linear dynamics of the process without bias correction.
        control_matrix (np.ndarray): Control input matrix B of shape (n_states, n_controls).
            Maps control actions to their expected effect on physical state evolution.
        initial_state_mean (np.ndarray): Initial estimate of physical states of shape (n_states,).
            Bias states are initialized to zero and learned during operation.
        process_noise_std (float, optional): Process noise standard deviation for physical states.
            Represents uncertainty in the nominal model dynamics. Default: 0.5
        measurement_noise_std (float, optional): Measurement noise standard deviation.
            Should reflect actual sensor accuracy in process units. Default: 10.0
        bias_process_noise_std (float, optional): Process noise for bias state evolution.
            Controls adaptation rate - smaller values mean slower bias learning. Default: 0.1
    
    Attributes:
        n_physical_states (int): Number of observable process states
        n_controls (int): Number of control inputs  
        n_augmented_states (int): Total augmented state dimension (2 * n_physical_states)
        original_A (np.ndarray): Stored nominal transition matrix
        original_B (np.ndarray): Stored nominal control matrix
        kf (KalmanFilter): Underlying pykalman filter with augmented state space
        
    Example:
        >>> # Configure for granulation process with suspected model bias
        >>> A = np.array([[0.9, 0.05], [0.0, 0.98]])
        >>> B = np.array([[0.3, -0.5], [0.0, 0.01]])
        >>> initial_state = np.array([400.0, 1.5])
        >>> estimator = BiasAugmentedKalmanStateEstimator(A, B, initial_state,
        ...                                               bias_process_noise_std=0.05)
        >>> 
        >>> # Process measurements and track bias learning
        >>> measurement = np.array([415.2, 1.48])
        >>> control = np.array([120.0, 500.0, 30.0])
        >>> physical_est, bias_est = estimator.estimate(measurement, control)
        >>> print(f"Learned bias: {bias_est}")  # Shows systematic offset correction
    
    Notes:
        - Bias states evolve as random walks, allowing slow adaptation to changing conditions
        - Observation matrix C_aug = [I, 0] ensures bias affects dynamics, not measurements
        - Computational complexity is approximately 8x that of standard Kalman filter
        - For known fixed bias, consider ProcessBiasKalmanEstimator for efficiency
        - Convergence time depends on bias_process_noise_std and process excitation
    """
    def __init__(self, transition_matrix, control_matrix, initial_state_mean, 
                 process_noise_std=0.5, measurement_noise_std=10.0, bias_process_noise_std=0.1):
        """Initialize bias-augmented Kalman filter with state space augmentation.
        
        Constructs the augmented state-space representation by extending the original
        system with bias states and configuring the Kalman filter matrices accordingly.
        The bias states are initialized to zero and will be learned online through
        the filtering process.
        
        Args:
            transition_matrix (np.ndarray): Physical state transition matrix A of shape (n_states, n_states).
                Defines nominal dynamics without bias correction.
            control_matrix (np.ndarray): Control input matrix B of shape (n_states, n_controls).
                Maps control actions to state evolution.
            initial_state_mean (np.ndarray): Initial physical state estimate of shape (n_states,).
                Should represent best available estimate of true initial process state.
            process_noise_std (float, optional): Process noise for physical state evolution.
                Larger values allow more deviation from nominal model. Default: 0.5
            measurement_noise_std (float, optional): Sensor measurement noise level.
                Should match actual sensor specifications. Default: 10.0
            bias_process_noise_std (float, optional): Bias state evolution noise.
                Controls bias adaptation rate - smaller means slower learning. Default: 0.1
        
        Raises:
            ValueError: If matrix dimensions are incompatible
            LinAlgError: If transition matrix is singular or ill-conditioned
        
        Notes:
            - Bias states are initialized to zero with high initial uncertainty
            - Physical state initial covariance is set conservatively
            - Augmented system dimension is 2 * n_physical_states
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
    """Kalman Filter with fixed process bias correction for pharmaceutical applications.
    
    This estimator implements a computationally efficient approach to process bias
    correction by incorporating a known systematic offset directly into the state
    evolution equation. Designed for scenarios where the bias term is well-characterized
    (e.g., from regression analysis) and remains approximately constant during operation.
    
    Mathematical Model (Fixed Process Bias):
        State evolution: x[k+1] = A*x[k] + B*u[k] + intercept + w[k]
        Observation:     y[k] = x[k] + v[k]
    
    Where the intercept term represents systematic model errors such as:
        - Missing intercept from sklearn linear regression models
        - Unmodeled steady-state offsets in process dynamics
        - Constant disturbances or unmeasured inputs
    
    Key Characteristics:
        - No state augmentation required (standard Kalman filter complexity)
        - Immediate bias correction from first time step
        - Deterministic bias handling (no online learning)
        - Optimal for time-invariant systematic errors
    
    Trade-offs:
        Advantages:
            - Computational efficiency (O(n³) vs O(8n³) for adaptive methods)
            - Immediate bias correction without learning period
            - Simple implementation and tuning
            - Mathematically equivalent to adaptive methods at convergence
        
        Limitations:
            - Requires accurate a priori knowledge of bias magnitude
            - Cannot adapt to changing process conditions
            - Performance degrades if bias estimate is inaccurate
            - Not suitable for time-varying or unknown biases
    
    Applications:
        - Pharmaceutical granulation with known sklearn regression intercepts
        - Process control where historical data provides bias estimates
        - Systems with well-characterized steady-state offsets
        - Computational efficiency is prioritized over adaptability
    
    Args:
        transition_matrix (np.ndarray): State transition matrix A of shape (n_states, n_states).
            Describes nominal process dynamics without bias terms.
        control_matrix (np.ndarray): Control input matrix B of shape (n_states, n_controls).
            Maps control actions to their effect on state evolution.
        initial_state_mean (np.ndarray): Initial true state estimate of shape (n_states,).
            Should represent the actual physical state without bias contamination.
        intercept_term (np.ndarray): Fixed process bias vector of shape (n_states,).
            Typically obtained from regression analysis or process identification.
            Units must match state variables (e.g., μm for particle size).
        process_noise_std (float, optional): Process noise standard deviation.
            Represents uncertainty in biased dynamics model. Default: 0.5
        measurement_noise_std (float, optional): Sensor measurement noise level.
            Should reflect actual sensor specifications in process units. Default: 10.0
    
    Attributes:
        n_states (int): Number of process states
        n_controls (int): Number of control inputs
        intercept (np.ndarray): Stored fixed bias term applied to dynamics
        transition_matrix (np.ndarray): Stored state transition matrix
        control_matrix (np.ndarray): Stored control input matrix
        kf (KalmanFilter): Standard Kalman filter instance
    
    Example:
        >>> # Configure for granulation with known regression intercept
        >>> A = np.array([[0.9, 0.05], [0.0, 0.98]])
        >>> B = np.array([[0.3, -0.5], [0.0, 0.01]])  
        >>> initial_state = np.array([400.0, 1.5])
        >>> intercept = np.array([26.3, 0.045])  # From sklearn fit
        >>> estimator = ProcessBiasKalmanEstimator(A, B, initial_state, intercept)
        >>> 
        >>> # Apply with immediate bias correction
        >>> measurement = np.array([415.2, 1.48])
        >>> control = np.array([120.0, 500.0, 30.0])
        >>> corrected_state = estimator.estimate(measurement, control)
    
    Notes:
        - Intercept is added to transition offset, not applied to measurements
        - Best performance when intercept accurately represents true systematic bias
        - Consider BiasAugmentedKalmanStateEstimator for unknown or varying bias
        - Computational cost identical to standard Kalman filter
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
    """Kalman Filter with fixed measurement bias correction for sensor calibration errors.
    
    This estimator addresses systematic sensor bias by pre-correcting measurements before
    applying standard Kalman filtering. Designed for scenarios where sensors have known
    calibration offsets but the underlying process dynamics are correctly modeled.
    
    Mathematical Model (Fixed Measurement Bias):
        State evolution: x[k+1] = A*x[k] + B*u[k] + w[k]    (unbiased dynamics)
        Raw observation: y_raw[k] = x[k] + bias + v[k]      (biased measurements)
        Corrected obs.:  y[k] = y_raw[k] - bias             (bias removal)
    
    The filter operates on bias-corrected measurements y[k], enabling accurate state
    estimation despite systematic sensor offsets. This approach is complementary to
    process bias correction and addresses different error sources.
    
    Key Characteristics:
        - Pre-processing approach: bias removed before filtering
        - Standard Kalman filter complexity (no augmentation)
        - Assumes perfect knowledge of measurement bias magnitude
        - Optimal for time-invariant sensor calibration errors
    
    Applications:
        - Pressure sensors with systematic calibration drift
        - Temperature measurements with thermocouple bias
        - Flow meters with consistent offset errors
        - Analytical instruments with zero-point drift
        - Any sensor with known, stable calibration offset
    
    Comparison with Process Bias:
        Measurement Bias: y = x + bias (sensor problem)
        Process Bias: x[k+1] = f(x[k]) + bias (model problem)
        
        Use measurement bias correction when:
            - Sensors are known to be mis-calibrated
            - Process model is accurate
            - Bias appears in measurements, not dynamics
    
    Args:
        transition_matrix (np.ndarray): State transition matrix A of shape (n_states, n_states).
            Should accurately represent true process dynamics.
        control_matrix (np.ndarray): Control input matrix B of shape (n_states, n_controls).
            Maps control actions to their effect on state evolution.
        initial_state_mean (np.ndarray): Initial true state estimate of shape (n_states,).
            Should represent actual physical state without measurement bias.
        intercept_term (np.ndarray): Fixed measurement bias vector of shape (n_states,).
            Systematic offset in sensor readings, typically from calibration analysis.
            Units must match measurement variables (e.g., μm, %, psi).
        process_noise_std (float, optional): Process noise standard deviation.
            Represents uncertainty in true dynamics model. Default: 0.5
        measurement_noise_std (float, optional): Sensor noise after bias correction.
            Random component of sensor error after systematic bias removal. Default: 10.0
    
    Attributes:
        n_states (int): Number of measured process states
        control_matrix (np.ndarray): Stored control input matrix
        intercept (np.ndarray): Fixed measurement bias vector applied to observations
        kf (KalmanFilter): Standard Kalman filter operating on corrected measurements
    
    Example:
        >>> # Configure for sensors with known calibration offset
        >>> A = np.array([[0.9, 0.05], [0.0, 0.98]])
        >>> B = np.array([[0.3, -0.5], [0.0, 0.01]])
        >>> initial_state = np.array([400.0, 1.5])  # True initial state
        >>> sensor_bias = np.array([15.2, -0.03])   # Known sensor offset
        >>> estimator = MeasurementBiasKalmanEstimator(A, B, initial_state, sensor_bias)
        >>> 
        >>> # Process biased measurement
        >>> biased_measurement = np.array([430.4, 1.52])  # Includes sensor bias
        >>> control = np.array([120.0, 500.0, 30.0])
        >>> true_state_est = estimator.estimate(biased_measurement, control)
    
    Notes:
        - Bias correction applied before Kalman update: y_corrected = y_raw - bias
        - Measurement noise represents post-correction sensor uncertainty
        - Best performance when bias accurately represents systematic sensor error
        - For unknown measurement bias, consider sensor recalibration first
        - Computational cost identical to standard Kalman filter
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