import numpy as np

class AdvancedPlantSimulator:
    """
    A more realistic simulator for the continuous granulation process, featuring
    nonlinear dynamics, state interactions, time lags, and disturbances.
    """
    def __init__(self, initial_state=None):
        if initial_state is None:
            self.state = {
                'd50': 400.0,  # Median particle size in micrometers
                'lod': 1.5,    # Loss on drying in percent
            }
        else:
            self.state = initial_state

        # --- Internal process variables ---
        # Lag buffers to simulate material transport time
        self.d50_lag_buffer = np.full(15, self.state['d50'])
        self.lod_lag_buffer = np.full(25, self.state['lod'])

        # Disturbance variable: filter blockage (starts at 0, slowly increases)
        self.filter_blockage = 0.0

    def _update_disturbance(self):
        """Simulates slow filter clogging over time."""
        # The filter gets slightly more clogged each time step, reducing drying efficiency
        self.filter_blockage += 0.0005 

    def step(self, cpps):
        """
        Updates the plant state for one time step based on the current CPPs.

        Args:
            cpps (dict): A dictionary of current Critical Process Parameters.
                         Keys: 'spray_rate', 'air_flow', 'carousel_speed'

        Returns:
            dict: The new state of the plant (d50, lod).
        """
        # Update the disturbance first
        self._update_disturbance()

        # === d50 (Granule Size) Dynamics ===
        # Effect of spray rate is nonlinear (saturates at high values)
        spray_effect = 150 * np.tanh((cpps['spray_rate'] - 120) / 40.0)
        # Effect of carousel speed (higher speed -> less time to agglomerate)
        speed_effect = - (cpps['carousel_speed'] - 30) * 5.0
        d50_target = 450 + spray_effect + speed_effect

        # Apply process lag using a moving average buffer
        self.d50_lag_buffer = np.roll(self.d50_lag_buffer, -1)
        self.d50_lag_buffer[-1] = d50_target
        # Add Gaussian noise to simulate measurement error
        current_d50 = np.mean(self.d50_lag_buffer) + np.random.normal(0, 5) 

        # === LOD (Moisture) Dynamics ===
        # Effect of air flow is dominant for drying
        air_flow_effect = - (cpps['air_flow'] - 500) * 0.008
        # Effect of carousel speed (less time in dryer -> higher LOD)
        drying_time_effect = (cpps['carousel_speed'] - 30) * 0.05

        # INTERACTION: Larger granules are harder to dry
        granule_size_effect = (current_d50 - 400) * 0.002

        # DISTURBANCE: Filter blockage reduces drying efficiency (increases LOD)
        disturbance_effect = self.filter_blockage

        lod_target = 2.0 + air_flow_effect + drying_time_effect + granule_size_effect + disturbance_effect

        # Apply process lag
        self.lod_lag_buffer = np.roll(self.lod_lag_buffer, -1)
        self.lod_lag_buffer[-1] = lod_target
        # Add noise
        current_lod = np.mean(self.lod_lag_buffer) + np.random.normal(0, 0.05)

        # Update the state and return
        self.state = {'d50': max(50, current_d50), 'lod': max(0.1, current_lod)}
        return self.state
