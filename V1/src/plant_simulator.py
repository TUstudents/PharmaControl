import numpy as np

class AdvancedPlantSimulator:
    """High-fidelity simulator for pharmaceutical continuous granulation processes.
    
    This simulator models the complex dynamics of a continuous granulation process
    with realistic features including:
    - Nonlinear process dynamics with saturation effects
    - Time delays representing material transport through equipment
    - Process interactions between particle size and moisture content
    - Unmeasured disturbances (filter blockage) affecting drying efficiency
    - Measurement noise simulating real sensor characteristics
    
    The process models two critical material attributes:
    - d50: Median particle size (micrometers) - affected by spray rate and mixing
    - LOD: Loss on drying/moisture content (%) - affected by air flow and residence time
    
    Attributes:
        state: Current process state dictionary containing 'd50' and 'lod' values
        d50_lag_buffer: Circular buffer simulating particle size transport delay (15 steps)
        lod_lag_buffer: Circular buffer simulating moisture transport delay (25 steps)
        filter_blockage: Unmeasured disturbance representing filter degradation over time
    
    Example:
        >>> simulator = AdvancedPlantSimulator()
        >>> cpps = {'spray_rate': 120.0, 'air_flow': 500.0, 'carousel_speed': 30.0}
        >>> new_state = simulator.step(cpps)
        >>> print(f"Particle size: {new_state['d50']:.1f} μm")
    """
    def __init__(self, initial_state=None):
        """Initialize the granulation process simulator with specified or default conditions.
        
        Args:
            initial_state: Optional dictionary specifying initial process conditions.
                Must contain keys 'd50' and 'lod' with numeric values.
                If None, uses default steady-state values:
                - d50: 400.0 μm (typical target particle size)
                - lod: 1.5% (typical target moisture content)
        
        Notes:
            - Transport delays are modeled using circular buffers initialized to steady-state
            - Filter blockage disturbance starts at zero (clean filter condition)
            - Buffer sizes (15 for d50, 25 for lod) represent realistic transport delays
        """
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
        """Update unmeasured disturbance representing progressive filter degradation.
        
        Models the gradual accumulation of particulate matter in air filtration
        systems, which reduces drying efficiency over time. This represents a
        realistic unmeasured disturbance that MPC systems must handle through
        integral action or adaptive mechanisms.
        
        Notes:
            - Blockage increases by 0.0005 per time step (0.5 per 1000 steps)
            - This creates a slow drift that challenges steady-state control
            - In practice, filters would be cleaned/replaced before significant impact
        """
        # The filter gets slightly more clogged each time step, reducing drying efficiency
        self.filter_blockage += 0.0005 

    def step(self, cpps):
        """Execute one simulation time step with specified control inputs.
        
        Advances the granulation process simulation by one discrete time step,
        computing new values for critical material attributes based on:
        - Current control parameter settings (CPPs)
        - Nonlinear process dynamics with realistic physics
        - Time delays from material transport through equipment
        - Process interactions between particle size and moisture
        - Progressive disturbance effects (filter blockage)
        - Measurement noise typical of industrial sensors
        
        The simulation implements validated process models including:
        - Spray rate saturation effects on particle agglomeration
        - Air flow impact on moisture removal efficiency  
        - Carousel speed effects on residence time and mixing
        - Cross-coupling between particle size and drying difficulty
        
        Args:
            cpps: Dictionary of critical process parameters with required keys:
                - 'spray_rate': Liquid binder spray rate (g/min), typically 80-180
                - 'air_flow': Fluidization air flow rate (m³/h), typically 400-700
                - 'carousel_speed': Carousel rotation speed (rpm), typically 20-40
        
        Returns:
            Updated process state dictionary containing:
            - 'd50': Median particle size (μm) with transport delay and noise
            - 'lod': Loss on drying moisture content (%) with delay and noise
            
            Values are bounded to physically realistic ranges (d50 ≥ 50 μm, lod ≥ 0.1%)
        
        Notes:
            - Time step represents ~1 second of real process time
            - Transport delays: 15 steps for particle size, 25 steps for moisture
            - Gaussian noise: σ=5 μm for d50, σ=0.05% for lod
            - Disturbance accumulates at 0.0005 per step affecting moisture removal
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
