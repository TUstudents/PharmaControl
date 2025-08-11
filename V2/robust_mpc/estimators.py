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
