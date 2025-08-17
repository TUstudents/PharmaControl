import numpy as np
import torch

class RobustMPCController:
    """Advanced Model Predictive Controller for robust pharmaceutical process control.
    
    This controller integrates multiple advanced techniques to achieve superior performance
    in pharmaceutical continuous granulation processes:
    
    1. **Robust State Estimation**: Bias-corrected Kalman filtering for accurate state estimation
    2. **Probabilistic Prediction**: Uncertainty quantification through dropout-based ensembles
    3. **Genetic Optimization**: Global optimization for complex, constrained control problems
    4. **Integral Action**: Disturbance estimation for offset-free tracking performance
    
    Architecture Components:
        - State Estimator: Handles sensor noise and systematic model bias
        - Probabilistic Model: Provides mean predictions with uncertainty bounds
        - Genetic Optimizer: Searches complex action spaces with constraints
        - Risk Management: Considers prediction uncertainty in control decisions
    
    Key Features:
        - Offset-free control through integral action and bias correction
        - Uncertainty-aware control decisions based on prediction confidence
        - Robust optimization using genetic algorithms for global optima
        - Multi-objective optimization balancing tracking performance and risk
    
    CRITICAL IMPLEMENTATION DETAIL - Data Scaling:
        This controller implements TWO DISTINCT scaling methods for different data types:
        
        1. **Value Scaling** (_scale_cma_vector, _scale_cma_plan, _scale_cpp_plan):
           - Formula: scaled = (value - min) / (max - min)
           - Used for: Absolute measurements, setpoints, control actions
           - Includes translation term (- min) to map range to [0,1]
           - Appropriate for scaling actual process measurements
        
        2. **Offset Scaling** (_scale_cma_offset):
           - Formula: scaled = offset / (max - min)  [NO translation]
           - Used for: Disturbance estimates, corrections, integral action terms
           - NO translation term - preserves zero-mean property
           - CRITICAL for proper integral action functionality
        
        Mathematical Justification:
            Offsets represent corrections/disturbances, not absolute values.
            Adding a constant (min) to a correction term fundamentally changes
            its meaning and breaks the integral action. The offset scaling 
            preserves proportionality while maintaining the zero-mean nature
            of disturbance estimates.
        
        Example Impact:
            - True disturbance: +30 μm in d50 (range 300-600 μm)
            - Value scaling: (30 - 300) / 300 = -0.9  [WRONG - negative bias!]
            - Offset scaling: 30 / 300 = 0.1           [CORRECT - preserves sign]
        
        This distinction is ESSENTIAL for offset-free MPC functionality.
    
    Args:
        model: Probabilistic neural network model implementing predict_distribution()
            Must provide both mean predictions and uncertainty estimates
        estimator: State estimator (KalmanStateEstimator or bias-corrected variant)
            Provides filtered state estimates from noisy sensor measurements
        optimizer_class: Genetic algorithm optimizer class for action optimization
            Should implement population-based optimization with constraint handling
        config (dict): Controller configuration containing:
            - 'cma_names': List of Critical Material Attribute names
            - 'cpp_names': List of Critical Process Parameter names  
            - 'horizon': Control and prediction horizon length
            - 'integral_gain': Adaptation rate for disturbance estimation
            - 'mc_samples': Number of Monte Carlo samples for uncertainty quantification
            - 'risk_aversion': Risk penalty weight (higher = more conservative)
            - 'verbose': Enable verbose validation logging (default: False)
        scalers (dict): Data scaling transformations for consistent preprocessing
            Should contain fitted scalers for all process variables
    
    Attributes:
        model: Stored probabilistic prediction model
        estimator: Stored state estimation module
        optimizer_class: Stored optimization algorithm class
        config (dict): Stored controller configuration
        scalers (dict): Stored data preprocessing scalers
        device (str): PyTorch device for model computations
        disturbance_estimate (np.ndarray): Current integral action term
    
    Example:
        >>> # Configure robust MPC for granulation process
        >>> controller = RobustMPCController(
        ...     model=probabilistic_transformer,
        ...     estimator=bias_corrected_kalman,
        ...     optimizer_class=GeneticOptimizer,
        ...     config=mpc_config,
        ...     scalers=data_scalers
        ... )
        >>> 
        >>> # Execute control step
        >>> past_cmas = df_history[['d50', 'lod']].values
        >>> past_cpps = df_history[['spray_rate', 'air_flow', 'carousel_speed']].values
        >>> targets = np.array([450.0, 1.8])  # Target d50=450μm, LOD=1.8%
        >>> optimal_action = controller.suggest_action(past_cmas, past_cpps, targets)
    
    Notes:
        - Automatically handles device placement for GPU acceleration when available
        - Integral action provides offset-free tracking for unmeasured disturbances
        - Risk-aware optimization considers both tracking error and prediction uncertainty
        - Genetic optimization enables handling of complex, non-convex constraint spaces
    """
    def __init__(self, model, estimator, optimizer_class, config, scalers):
        self.model = model
        self.estimator = estimator
        self.optimizer_class = optimizer_class
        self.config = config
        self.scalers = scalers
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        # Validate configuration and scalers
        self._validate_initialization()

        # Initialize the disturbance estimate for Integral Action
        self.disturbance_estimate = np.zeros(len(config['cma_names']))

    def _update_disturbance_estimate(self, smooth_state, setpoint):
        """Updates the integral error term for offset-free control."""
        error = setpoint - smooth_state
        # The gain (alpha) determines how quickly the controller adapts to the disturbance
        alpha = self.config.get('integral_gain', 0.1)
        self.disturbance_estimate += alpha * error

    def _get_fitness_function(self, past_cmas_scaled, past_cpps_scaled, target_cmas_unscaled):
        """
        Creates and returns the fitness function to be used by the GA.
        This function captures the current state and target.
        """
        def fitness(control_plan_unscaled):
            # --- 1. Prepare Inputs ---
            # Scale the unscaled control plan generated by the GA
            plan_scaled = self._scale_cpp_plan(control_plan_unscaled)

            # Convert all inputs to tensors
            past_cmas_tensor = torch.tensor(past_cmas_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
            past_cpps_tensor = torch.tensor(past_cpps_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
            future_cpps_tensor = torch.tensor(plan_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)

            # --- 2. Get Probabilistic Prediction ---
            mean_pred_scaled, std_pred_scaled = self.model.predict_distribution(
                past_cmas_tensor, past_cpps_tensor, future_cpps_tensor, 
                n_samples=self.config.get('mc_samples', 30)
            )

            # --- 3. Calculate Risk-Adjusted, Corrected Prediction ---
            # Correct for disturbance (Integral Action)
            # CRITICAL: Use offset scaling for disturbance estimates, not value scaling
            disturbance_scaled = self._scale_cma_offset(self.disturbance_estimate)
            corrected_mean_scaled = mean_pred_scaled + torch.tensor(disturbance_scaled, device=self.device)

            # Adjust for risk (Uncertainty-Awareness)
            beta = self.config.get('risk_beta', 1.5) # Higher beta = more cautious
            # For minimization, we penalize the upper bound of the error
            risk_adjusted_pred_scaled = corrected_mean_scaled + beta * std_pred_scaled

            # --- 4. Calculate Cost ---
            target_scaled = self._scale_cma_plan(target_cmas_unscaled)
            target_tensor = torch.tensor(target_scaled, dtype=torch.float32).to(self.device)
            cost = torch.mean(torch.abs(risk_adjusted_pred_scaled - target_tensor))

            return cost.item()

        return fitness

    def suggest_action(self, noisy_measurement, control_input, setpoint):
        # 1. Get a clean state estimate
        smooth_state = self.estimator.estimate(noisy_measurement, control_input)

        # 2. Update the integral error term
        self._update_disturbance_estimate(smooth_state, setpoint)

        # 3. Create the fitness function for this specific time step
        # This part requires getting the historical data, which we will simulate for the test.
        # In a real app, this would come from a data buffer.
        past_cmas_scaled, past_cpps_scaled = self._get_scaled_history(smooth_state)

        # The target is the setpoint repeated over the horizon
        target_plan = np.tile(setpoint, (self.config['horizon'], 1))
        fitness_func = self._get_fitness_function(past_cmas_scaled, past_cpps_scaled, target_plan)

        # 4. Instantiate and run the optimizer
        param_bounds = self._get_param_bounds()
        optimizer = self.optimizer_class(fitness_func, param_bounds, self.config['ga_config'])
        best_plan = optimizer.optimize()

        # 5. Return the first step of the optimal plan
        return best_plan[0]

    # --- Helper methods for scaling and data management ---
    def _get_scaled_history(self, current_smooth_state):
        """Get scaled historical data for model prediction.
        
        In a production system, this would pull from a rolling data buffer.
        For testing, we create reasonable historical data based on current state.
        
        Args:
            current_smooth_state (np.ndarray): Current filtered CMA state
            
        Returns:
            tuple: (past_cmas_scaled, past_cpps_scaled) for model input
        """
        L = self.config['lookback']
        
        # Create reasonable CMA history with some variation around current state
        # This simulates a process that has been operating near the current conditions
        noise_scale = 0.05  # 5% variation in historical data
        past_cmas_unscaled = np.zeros((L, len(current_smooth_state)))
        
        for i in range(L):
            # Add small random variations to simulate realistic historical trajectory
            variation = np.random.normal(0, noise_scale, size=current_smooth_state.shape)
            past_cmas_unscaled[i] = current_smooth_state * (1 + variation)
            
        # Create reasonable CPP history for pharmaceutical granulation process
        # Use typical operating conditions as baseline
        baseline_cpps = {
            'spray_rate': 130.0,      # g/min - typical spray rate
            'air_flow': 550.0,        # m³/h - typical air flow  
            'carousel_speed': 30.0    # rpm - typical carousel speed
        }
        
        past_cpps_unscaled = np.zeros((L, len(self.config['cpp_names'])))
        
        for i in range(L):
            for j, cpp_name in enumerate(self.config['cpp_names']):
                if cpp_name in baseline_cpps:
                    # Add process noise to baseline values
                    baseline = baseline_cpps[cpp_name]
                    variation = np.random.normal(0, noise_scale)
                    past_cpps_unscaled[i, j] = baseline * (1 + variation)
                    
                    # Ensure values stay within reasonable process limits
                    constraints = self.config.get('cpp_constraints', {})
                    if cpp_name in constraints:
                        min_val = constraints[cpp_name]['min_val']
                        max_val = constraints[cpp_name]['max_val'] 
                        past_cpps_unscaled[i, j] = np.clip(past_cpps_unscaled[i, j], min_val, max_val)
        
        # Scale both CMA and CPP historical data
        past_cmas_scaled = self._scale_cma_plan(past_cmas_unscaled)
        past_cpps_scaled = self._scale_cpp_plan(past_cpps_unscaled, with_soft_sensors=True)
        
        return past_cmas_scaled, past_cpps_scaled

    def _scale_cpp_plan(self, plan_unscaled, with_soft_sensors=False):
        """Scale CPP control plan using fitted scalers, optionally adding soft sensors.
        
        Args:
            plan_unscaled (np.ndarray): Unscaled CPP values, shape (horizon, num_cpps)
            with_soft_sensors (bool): Whether to add soft sensor features
            
        Returns:
            np.ndarray: Scaled CPP plan with optional soft sensors
        """
        self._validate_scaling_inputs(plan_unscaled, "CPP plan", "_scale_cpp_plan")
        
        if plan_unscaled.ndim == 1:
            plan_unscaled = plan_unscaled.reshape(1, -1)
            
        horizon, num_cpps = plan_unscaled.shape
        
        if with_soft_sensors:
            # Create extended array with soft sensors
            cpp_full_names = self.config['cpp_full_names']
            plan_with_sensors = np.zeros((horizon, len(cpp_full_names)))
            
            # Copy basic CPPs
            for i, cpp_name in enumerate(self.config['cpp_names']):
                if i < num_cpps:
                    plan_with_sensors[:, cpp_full_names.index(cpp_name)] = plan_unscaled[:, i]
            
            # Calculate soft sensors using pharmaceutical process physics
            spray_rate_idx = cpp_full_names.index('spray_rate')
            carousel_speed_idx = cpp_full_names.index('carousel_speed')
            specific_energy_idx = cpp_full_names.index('specific_energy')
            froude_number_idx = cpp_full_names.index('froude_number_proxy')
            
            spray_rate = plan_with_sensors[:, spray_rate_idx]
            carousel_speed = plan_with_sensors[:, carousel_speed_idx]
            
            # Specific energy: normalized spray rate × carousel speed interaction
            plan_with_sensors[:, specific_energy_idx] = (spray_rate * carousel_speed) / 1000.0
            
            # Froude number proxy: dimensionless mixing intensity measure
            plan_with_sensors[:, froude_number_idx] = (carousel_speed ** 2) / 9.81
            
            # Scale all features using fitted scalers
            plan_scaled = np.zeros_like(plan_with_sensors)
            for i, col_name in enumerate(cpp_full_names):
                if col_name in self.scalers:
                    # Reshape for scaler (expects 2D)
                    col_data = plan_with_sensors[:, i].reshape(-1, 1)
                    plan_scaled[:, i] = self.scalers[col_name].transform(col_data).flatten()
                else:
                    raise ValueError(f"Scaler for '{col_name}' not found in scalers dict")
            
            return plan_scaled
        else:
            # Scale basic CPPs only
            plan_scaled = np.zeros_like(plan_unscaled)
            for i, col_name in enumerate(self.config['cpp_names']):
                if i < num_cpps and col_name in self.scalers:
                    col_data = plan_unscaled[:, i].reshape(-1, 1)
                    plan_scaled[:, i] = self.scalers[col_name].transform(col_data).flatten()
                else:
                    raise ValueError(f"Scaler for '{col_name}' not found in scalers dict")
            
            return plan_scaled

    def _scale_cma_plan(self, plan_unscaled):
        """Scale CMA plan/target using fitted scalers.
        
        Args:
            plan_unscaled (np.ndarray): Unscaled CMA values, shape (horizon, num_cmas) or (num_cmas,)
            
        Returns:
            np.ndarray: Scaled CMA plan with same shape as input
        """
        self._validate_scaling_inputs(plan_unscaled, "CMA plan", "_scale_cma_plan")
        
        original_shape = plan_unscaled.shape
        
        # Handle both 1D vectors and 2D plans
        if plan_unscaled.ndim == 1:
            plan_unscaled = plan_unscaled.reshape(1, -1)
            
        horizon, num_cmas = plan_unscaled.shape
        plan_scaled = np.zeros_like(plan_unscaled)
        
        # Scale each CMA column using fitted scalers
        for i, col_name in enumerate(self.config['cma_names']):
            if i < num_cmas and col_name in self.scalers:
                # Reshape for scaler (expects 2D)
                col_data = plan_unscaled[:, i].reshape(-1, 1)
                plan_scaled[:, i] = self.scalers[col_name].transform(col_data).flatten()
            else:
                raise ValueError(f"Scaler for CMA '{col_name}' not found in scalers dict")
        
        # Return in original shape
        if len(original_shape) == 1:
            return plan_scaled.flatten()
        else:
            return plan_scaled

    def _scale_cma_vector(self, vector_unscaled):
        """Scale single CMA vector using fitted scalers (for absolute values, not offsets).
        
        Args:
            vector_unscaled (np.ndarray): Unscaled CMA vector, shape (num_cmas,)
            
        Returns:
            np.ndarray: Scaled CMA vector with same shape
        """
        self._validate_scaling_inputs(vector_unscaled, "CMA vector", "_scale_cma_vector")
        
        if vector_unscaled.ndim != 1:
            raise ValueError(f"Expected 1D vector, got shape {vector_unscaled.shape}")
            
        vector_scaled = np.zeros_like(vector_unscaled)
        
        # Scale each CMA element using fitted scalers
        for i, col_name in enumerate(self.config['cma_names']):
            if i < len(vector_unscaled) and col_name in self.scalers:
                # Reshape for scaler (expects 2D)
                value_reshaped = vector_unscaled[i].reshape(1, -1)
                vector_scaled[i] = self.scalers[col_name].transform(value_reshaped).flatten()[0]
            else:
                raise ValueError(f"Scaler for CMA '{col_name}' not found in scalers dict")
        
        return vector_scaled

    def _scale_cma_offset(self, offset_unscaled):
        """Scale CMA offset vector for integral action using only the scale factor.
        
        CRITICAL: Offsets represent corrections/disturbances, not absolute values.
        Unlike absolute values, offsets should NOT be translated by the scaler's minimum.
        
        Mathematical Foundation:
            - MinMaxScaler: scaled_value = (value - min) / (max - min)
            - For offsets: scaled_offset = offset / (max - min)  [NO translation]
            - This preserves the offset's zero-mean property and correct magnitude
        
        Args:
            offset_unscaled (np.ndarray): Unscaled CMA offset vector, shape (num_cmas,)
                Represents disturbance estimates or integral action corrections
            
        Returns:
            np.ndarray: Correctly scaled offset vector with same shape
            
        Example:
            >>> # Disturbance estimate of +50 μm in d50, +0.2% in LOD
            >>> offset = np.array([50.0, 0.2])
            >>> scaled_offset = controller._scale_cma_offset(offset)
            >>> # Result preserves proportional correction without bias
            
        Notes:
            - Essential for proper offset-free MPC functionality
            - Incorrect scaling breaks integral action and causes steady-state error
            - Only uses scaler.scale_ (1/(max-min)), not data_min_ translation
        """
        self._validate_scaling_inputs(offset_unscaled, "CMA offset", "_scale_cma_offset")
        
        if offset_unscaled.ndim != 1:
            raise ValueError(f"Expected 1D offset vector, got shape {offset_unscaled.shape}")
            
        offset_scaled = np.zeros_like(offset_unscaled)
        
        # Scale each CMA offset using only the range factor (no translation)
        for i, col_name in enumerate(self.config['cma_names']):
            if i < len(offset_unscaled) and col_name in self.scalers:
                scaler = self.scalers[col_name]
                # CRITICAL: Use only scale factor, no translation for offsets
                # scale_[0] = 1 / (max - min) from the fitted MinMaxScaler
                scale_factor = scaler.scale_[0]
                offset_scaled[i] = offset_unscaled[i] * scale_factor
            else:
                raise ValueError(f"Scaler for CMA '{col_name}' not found in scalers dict")
        
        return offset_scaled

    def _get_param_bounds(self):
        param_bounds = []
        cpp_config = self.config['cpp_constraints']
        for _ in range(self.config['horizon']):
            for name in self.config['cpp_names']:
                param_bounds.append((cpp_config[name]['min_val'], cpp_config[name]['max_val']))
        return param_bounds

    def _validate_initialization(self):
        """Validate controller configuration and scalers during initialization."""
        # Check required configuration keys
        required_keys = ['cma_names', 'cpp_names', 'cpp_full_names', 'horizon', 'lookback']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: '{key}'")
        
        # Validate CMA configuration
        if not isinstance(self.config['cma_names'], list) or len(self.config['cma_names']) == 0:
            raise ValueError("'cma_names' must be a non-empty list")
            
        # Validate CPP configuration
        if not isinstance(self.config['cpp_names'], list) or len(self.config['cpp_names']) == 0:
            raise ValueError("'cpp_names' must be a non-empty list")
            
        if not isinstance(self.config['cpp_full_names'], list) or len(self.config['cpp_full_names']) == 0:
            raise ValueError("'cpp_full_names' must be a non-empty list")
        
        # Validate that all basic CPPs are included in full CPP list
        for cpp_name in self.config['cpp_names']:
            if cpp_name not in self.config['cpp_full_names']:
                raise ValueError(f"CPP '{cpp_name}' not found in cpp_full_names")
        
        # Validate horizon and lookback
        if self.config['horizon'] <= 0:
            raise ValueError("'horizon' must be positive")
        if self.config['lookback'] <= 0:
            raise ValueError("'lookback' must be positive")
            
        # Validate scalers availability
        if not isinstance(self.scalers, dict):
            raise ValueError("'scalers' must be a dictionary")
            
        # Check that all required scalers are available
        required_scalers = self.config['cma_names'] + self.config['cpp_full_names']
        missing_scalers = []
        
        for scaler_name in required_scalers:
            if scaler_name not in self.scalers:
                missing_scalers.append(scaler_name)
                
        if missing_scalers:
            raise ValueError(f"Missing scalers for: {missing_scalers}")
            
        # Validate that scalers have required methods
        for scaler_name, scaler in self.scalers.items():
            if not hasattr(scaler, 'transform'):
                raise ValueError(f"Scaler for '{scaler_name}' missing 'transform' method")
                
        if self.config.get('verbose', False):
            print("RobustMPCController validation passed")
            print(f"  - CMAs: {self.config['cma_names']}")
            print(f"  - CPPs: {self.config['cpp_names']}")  
            print(f"  - Full CPPs: {self.config['cpp_full_names']}")
            print(f"  - Horizon: {self.config['horizon']}, Lookback: {self.config['lookback']}")
            print(f"  - Available scalers: {len(self.scalers)}")

    def _validate_scaling_inputs(self, data, expected_shape_desc, method_name):
        """Validate inputs to scaling methods."""
        if not isinstance(data, np.ndarray):
            raise TypeError(f"{method_name}: Expected numpy array, got {type(data)}")
            
        if data.size == 0:
            raise ValueError(f"{method_name}: Input array is empty")
            
        if not np.all(np.isfinite(data)):
            raise ValueError(f"{method_name}: Input contains non-finite values (NaN/inf)")
            
        if self.config.get('verbose', False):
            print(f"{method_name} input validation passed: shape {data.shape} ({expected_shape_desc})")
