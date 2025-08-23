import numpy as np
import torch
import pandas as pd
from .data_buffer import DataBuffer, StartupHistoryGenerator

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
        - Industrial-grade error handling with safe fallback strategies
    
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
        
        # Pre-initialize fallback control action with guaranteed safe defaults
        # CRITICAL: Ensures safe fallback is available from very first control step
        self._last_successful_action = self._calculate_safe_default_action()
        
        # Initialize setpoint tracking for intelligent optimizer reset
        self._last_setpoint = None
        
        # Initialize rolling history buffer for real trajectory tracking
        buffer_size = config.get('history_buffer_size', max(100, 3 * config['lookback']))
        self.history_buffer = DataBuffer(
            cma_features=len(config['cma_names']),
            cpp_features=len(config['cpp_names']),
            buffer_size=buffer_size,
            validate_sequence=True
        )
        
        # Startup history generator for initial operation
        self.startup_generator = None
        self._initialization_complete = False
        
        # Initialize optimizer as instance variable for efficiency
        # This prevents creating new DEAP populations on every control step
        if optimizer_class is not None:
            param_bounds = self._get_param_bounds()
            # Create complete GA config with required parameters
            ga_config = config['ga_config'].copy()
            ga_config['horizon'] = config['horizon']
            ga_config['num_cpps'] = len(config['cpp_names'])
            self.optimizer = optimizer_class(param_bounds, ga_config)
        else:
            self.optimizer = None

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
        
        CRITICAL: Expects scaled control plans from GA since _get_param_bounds()
        now returns scaled bounds. This ensures consistency in the optimization space.
        """
        def fitness(control_plan_scaled):
            # --- 1. Prepare Inputs ---
            # GA now provides scaled control plan directly (no scaling needed)
            plan_scaled = control_plan_scaled.copy()  # Use scaled plan directly

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
            beta = self.config.get('risk_beta', 1.0) # FIXED: Reduced from 1.5 to 1.0 for balanced exploration
            # For minimization, we penalize the upper bound of the error
            risk_adjusted_pred_scaled = corrected_mean_scaled + beta * std_pred_scaled

            # --- 4. Calculate Cost ---
            target_scaled = self._scale_cma_plan(target_cmas_unscaled)
            target_tensor = torch.tensor(target_scaled, dtype=torch.float32).to(self.device)
            cost = torch.mean(torch.abs(risk_adjusted_pred_scaled - target_tensor))

            return cost.item()

        return fitness

    def suggest_action(self, noisy_measurement, control_input, setpoint, timestamp=None):
        """Enhanced suggest_action with real history buffer integration.
        
        Args:
            noisy_measurement (np.ndarray): Current CMA measurements with sensor noise
            control_input (np.ndarray): Current CPP control inputs  
            setpoint (np.ndarray): Target CMA setpoints
            timestamp (float, optional): Unix timestamp for sequencing. If None, uses current time.
            
        Returns:
            np.ndarray: Optimal control action for next time step
        """
        # 0. CRITICAL SAFETY CHECK: Validate buffer integrity for pharmaceutical manufacturing
        if len(self.history_buffer) > 0:
            cma_len = len(self.history_buffer._cma_buffer)
            cpp_len = len(self.history_buffer._cpp_buffer)
            if cma_len != cpp_len:
                if self.config.get('verbose', False):
                    print(f"WARNING: Buffer misalignment detected - CMA: {cma_len}, CPP: {cpp_len}")
        
        # 1. Get a clean state estimate
        smooth_state = self.estimator.estimate(noisy_measurement, control_input)

        # 2. Update history buffer with real data (critical for accurate predictions)
        # CRITICAL SAFETY FIX: Use atomic operation to prevent race condition data misalignment
        try:
            self.history_buffer.add_sample(smooth_state, control_input, timestamp)
        except Exception as e:
            if self.config.get('verbose', False):
                print(f"Warning: Failed to update history buffer: {e}")
                
        # 2b. CRITICAL SAFETY CHECK: Validate buffer integrity for pharmaceutical manufacturing
        if len(self.history_buffer) > 0:
            stats = self.history_buffer.get_statistics()
            cma_len = stats['cma_samples']
            cpp_len = stats['cpp_samples']
            if cma_len != cpp_len:
                if self.config.get('verbose', False):
                    print(f"CRITICAL: Buffer misalignment detected - CMA: {cma_len}, CPP: {cpp_len}")
                    print("This indicates race condition corruption. Using fallback control.")
                return self._get_fallback_action(control_input)

        # 3. Update the integral error term
        self._update_disturbance_estimate(smooth_state, setpoint)

        # 4. Intelligent optimizer reset on significant setpoint changes
        if self._should_reset_optimizer(setpoint):
            if self.config.get('verbose', False):
                print(f"Significant setpoint change detected, resetting optimizer for fresh exploration")
            self._reset_optimizer()
        
        # Update setpoint tracking for next comparison
        self._last_setpoint = setpoint.copy()

        # 5. Get historical data for model prediction (real or startup)
        past_cmas_scaled, past_cpps_scaled = self._get_real_history()

        # 6. Create the fitness function for this specific time step
        target_plan = np.tile(setpoint, (self.config['horizon'], 1))
        fitness_func = self._get_fitness_function(past_cmas_scaled, past_cpps_scaled, target_plan)

        # 7. Use existing optimizer instance with current fitness function
        if self.optimizer is None:
            if self.config.get('verbose', False):
                print("No optimizer available, using fallback control strategy")
            return self._get_fallback_action(control_input)
            
        try:
            best_plan_scaled = self.optimizer.optimize(fitness_func)
            
            # Validate optimization result
            if best_plan_scaled is None or best_plan_scaled.size == 0:
                raise ValueError("Optimizer returned invalid result")
            
            # CRITICAL: Unscale the optimizer result back to physical units
            # Since optimizer now works in scaled space, we need to convert back
            best_plan_unscaled = self._unscale_cpp_plan(best_plan_scaled)
                
            # Store successful action for fallback (in physical units)
            self._last_successful_action = best_plan_unscaled[0].copy()
            
            # 8. Return the first step of the optimal plan (in physical units)
            return best_plan_unscaled[0]
            
        except Exception as e:
            if self.config.get('verbose', False):
                print(f"Optimizer failed: {e}")
                print("Using fallback control strategy")
            
            # Fallback strategy: return safe control action
            return self._get_fallback_action(control_input)

    # --- Helper methods for scaling and data management ---
    def _get_real_history(self):
        """Get real historical data from buffer or startup generator.
        
        This method replaces the previous mock history generation with actual
        trajectory data, critical for accurate model predictions in MPC.
        
        Returns:
            tuple: (past_cmas_scaled, past_cpps_scaled) for model input
            
        Raises:
            ValueError: If unable to provide sufficient historical data
        """
        lookback = self.config['lookback']
        
        # Check if we have sufficient real data in buffer
        if self.history_buffer.is_ready(lookback):
            # Use real historical data
            past_cmas_unscaled, past_cpps_unscaled = self.history_buffer.get_model_inputs(lookback)
            
            if self.config.get('verbose', False) and not self._initialization_complete:
                print("Switched to real history data for MPC predictions")
                self._initialization_complete = True
                
        else:
            # Use startup generator during initial operation
            available_samples = len(self.history_buffer)
            
            if available_samples > 0:
                # Partial real data available - combine with startup generation
                real_cmas, real_cpps = self.history_buffer.get_model_inputs(available_samples)
                
                # Initialize startup generator if needed
                if self.startup_generator is None:
                    latest_cma, latest_cpp, _ = self.history_buffer.get_latest()
                    if latest_cma is not None and latest_cpp is not None:
                        self.startup_generator = StartupHistoryGenerator(
                            cma_features=len(self.config['cma_names']),
                            cpp_features=len(self.config['cpp_names']),
                            initial_cma_state=latest_cma,
                            initial_cpp_state=latest_cpp
                        )
                
                # Generate startup history for missing samples
                missing_samples = lookback - available_samples
                startup_cmas, startup_cpps = self.startup_generator.generate_startup_history(missing_samples)
                
                # Combine startup and real data
                past_cmas_unscaled = np.vstack([startup_cmas, real_cmas])
                past_cpps_unscaled = np.vstack([startup_cpps, real_cpps])
                
            else:
                # No real data yet - use pure startup generation
                if self.startup_generator is None:
                    # Create default startup generator with safe pharmaceutical baselines
                    default_cma_state = np.array([450.0, 1.8])  # d50=450μm, LOD=1.8%
                    default_cpp_state = np.array([130.0, 550.0, 30.0])  # Safe baselines
                    
                    # Trim to actual feature counts
                    default_cma_state = default_cma_state[:len(self.config['cma_names'])]
                    default_cpp_state = default_cpp_state[:len(self.config['cpp_names'])]
                    
                    self.startup_generator = StartupHistoryGenerator(
                        cma_features=len(self.config['cma_names']),
                        cpp_features=len(self.config['cpp_names']),
                        initial_cma_state=default_cma_state,
                        initial_cpp_state=default_cpp_state
                    )
                
                past_cmas_unscaled, past_cpps_unscaled = self.startup_generator.generate_startup_history(lookback)
                
                if self.config.get('verbose', False):
                    print(f"Using startup history generation (lookback={lookback})")
        
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
            # Create robust DataFrame-based soft sensor calculation
            cpp_full_names = self.config['cpp_full_names']
            
            # Validate required variables for soft sensor calculations
            required_base_vars = ['spray_rate', 'carousel_speed']
            required_soft_vars = ['specific_energy', 'froude_number_proxy']
            
            missing_base = [var for var in required_base_vars if var not in cpp_full_names]
            missing_soft = [var for var in required_soft_vars if var not in cpp_full_names]
            
            if missing_base:
                raise ValueError(f"Missing required base variables for soft sensors: {missing_base}")
            if missing_soft:
                raise ValueError(f"Missing required soft sensor variables: {missing_soft}")
            
            # Initialize DataFrame with all cpp_full_names columns
            plan_df = pd.DataFrame(
                data=np.zeros((horizon, len(cpp_full_names))), 
                columns=cpp_full_names
            )
            
            # Copy basic CPPs by name (robust to column order changes)
            for i, cpp_name in enumerate(self.config['cpp_names']):
                if i < num_cpps:
                    if cpp_name not in cpp_full_names:
                        raise ValueError(f"CPP '{cpp_name}' not found in cpp_full_names")
                    plan_df[cpp_name] = plan_unscaled[:, i]
            
            # Calculate soft sensors using robust column-name-based approach
            # CRITICAL: This approach is immune to column order changes
            try:
                # Specific energy: normalized spray rate × carousel speed interaction
                plan_df['specific_energy'] = (plan_df['spray_rate'] * plan_df['carousel_speed']) / 1000.0
                
                # Froude number proxy: dimensionless mixing intensity measure  
                plan_df['froude_number_proxy'] = (plan_df['carousel_speed'] ** 2) / 9.81
                
            except KeyError as e:
                raise ValueError(f"Soft sensor calculation failed - missing column: {e}")
            
            # Convert back to numpy array for scaling
            plan_with_sensors = plan_df.values
            
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
                # Convert scalar to 2D array for scaler (expects 2D)
                value_reshaped = np.array([[vector_unscaled[i]]])
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

    def _unscale_cpp_plan(self, plan_scaled):
        """Unscale CPP control plan from scaled space back to physical units.
        
        This method reverses the MinMaxScaler transformation to convert scaled
        control plans (from GeneticOptimizer) back to physical units.
        
        Args:
            plan_scaled (np.ndarray): Scaled CPP values, shape (horizon, num_cpps)
                with values in [0,1] range from MinMaxScaler
            
        Returns:
            np.ndarray: Unscaled CPP plan in physical units (g/min, m³/h, rpm, etc.)
            
        Raises:
            ValueError: If scalers are missing for any CPP parameter
            
        Notes:
            - Uses inverse_transform() method of fitted MinMaxScalers
            - Essential for converting optimizer output to physical control actions
            - Only handles basic CPPs, not soft sensors
        """
        self._validate_scaling_inputs(plan_scaled, "Scaled CPP plan", "_unscale_cpp_plan")
        
        if plan_scaled.ndim == 1:
            plan_scaled = plan_scaled.reshape(1, -1)
            
        horizon, num_cpps = plan_scaled.shape
        plan_unscaled = np.zeros_like(plan_scaled)
        
        # Unscale each CPP column using fitted scalers
        for i, col_name in enumerate(self.config['cpp_names']):
            if i < num_cpps and col_name in self.scalers:
                # Reshape for scaler (expects 2D)
                col_data = plan_scaled[:, i].reshape(-1, 1)
                plan_unscaled[:, i] = self.scalers[col_name].inverse_transform(col_data).flatten()
            else:
                raise ValueError(f"Scaler for CPP '{col_name}' not found in scalers dict")
        
        return plan_unscaled

    def _get_param_bounds(self):
        """Get scaled parameter bounds for GeneticOptimizer.
        
        CRITICAL: Returns bounds in scaled space [0,1] to match the fitness function's
        expectation that the optimizer provides scaled control plans.
        
        Returns:
            list: Scaled parameter bounds [(min_scaled, max_scaled), ...] for each
                parameter in the control horizon. All bounds are in [0,1] range.
                
        Raises:
            ValueError: If scalers are missing for any CPP parameter
            
        Notes:
            - GeneticOptimizer works entirely in scaled space for consistency
            - Fitness function expects scaled control plans as input
            - Bounds correspond to horizon × num_cpps parameters
        """
        param_bounds = []
        cpp_config = self.config['cpp_constraints']
        
        for _ in range(self.config['horizon']):
            for name in self.config['cpp_names']:
                if name not in self.scalers:
                    raise ValueError(f"Scaler for CPP '{name}' not found in scalers dict")
                
                # Get unscaled constraint bounds
                min_val = cpp_config[name]['min_val']
                max_val = cpp_config[name]['max_val']
                
                # Transform to scaled space using fitted scaler
                scaler = self.scalers[name]
                min_scaled = scaler.transform([[min_val]])[0, 0]
                max_scaled = scaler.transform([[max_val]])[0, 0]
                
                # Ensure proper ordering (min <= max)
                if min_scaled > max_scaled:
                    min_scaled, max_scaled = max_scaled, min_scaled
                
                param_bounds.append((min_scaled, max_scaled))
                
        return param_bounds
    
    def _calculate_safe_default_action(self):
        """Calculate safe default control action using constraint midpoints.
        
        This method provides guaranteed safe control values by using the midpoint
        of each parameter's constraint bounds. These values are always within 
        operational limits and provide a conservative, stable operating point
        for pharmaceutical process control.
        
        Returns:
            np.ndarray: Safe control action with constraint midpoint values
            
        Raises:
            ValueError: If required constraints are missing or invalid
            
        Notes:
            - Called during initialization to ensure safe fallback always available
            - Uses constraint midpoints as conservative safe operating point
            - Essential for pharmaceutical manufacturing safety during startup
        """
        safe_action = np.zeros(len(self.config['cpp_names']))
        cpp_config = self.config['cpp_constraints']
        
        for i, name in enumerate(self.config['cpp_names']):
            if name in cpp_config:
                min_val = cpp_config[name]['min_val']
                max_val = cpp_config[name]['max_val']
                
                # Validate constraint bounds
                if min_val >= max_val:
                    raise ValueError(f"Invalid constraint bounds for '{name}': min_val={min_val} >= max_val={max_val}")
                
                # Use conservative midpoint as safe default
                safe_action[i] = (min_val + max_val) / 2.0
            else:
                raise ValueError(f"Missing constraint configuration for CPP '{name}' - required for safe fallback")
                
        return safe_action
    
    def _should_reset_optimizer(self, current_setpoint):
        """Determine if optimizer should be reset due to significant setpoint change.
        
        Detects when setpoint changes are large enough to warrant fresh GA population
        exploration rather than continuing with existing population bias.
        
        Args:
            current_setpoint (np.ndarray): Current target setpoint values
            
        Returns:
            bool: True if optimizer should be reset for fresh exploration
            
        Notes:
            - Uses configurable threshold for change detection
            - Supports both absolute and relative change metrics
            - Essential for pharmaceutical grade transitions
        """
        # Skip reset if feature disabled
        if not self.config.get('reset_optimizer_on_setpoint_change', True):
            return False
            
        # Always reset on first setpoint (no previous reference)
        if self._last_setpoint is None:
            return False  # Don't reset on very first call
            
        # Calculate setpoint change magnitude
        setpoint_change = np.abs(current_setpoint - self._last_setpoint)
        
        # Get configured threshold (default: 5% relative change)
        threshold = self.config.get('setpoint_change_threshold', 0.05)
        
        # Use relative change detection for pharmaceutical applications
        if np.any(setpoint_change / (np.abs(self._last_setpoint) + 1e-6) > threshold):
            return True
            
        return False
    
    def _reset_optimizer(self):
        """Reset optimizer to fresh state for new exploration.
        
        Reinitializes the genetic algorithm optimizer with fresh random population
        to avoid population bias when setpoints change significantly.
        
        Notes:
            - Called automatically on significant setpoint changes
            - Maintains same parameter bounds and GA configuration
            - Essential for pharmaceutical grade transition control
        """
        if self.optimizer is not None and self.optimizer_class is not None:
            # Reinitialize with same bounds and configuration
            param_bounds = self._get_param_bounds()
            ga_config = self.config['ga_config'].copy()
            ga_config['horizon'] = self.config['horizon']
            ga_config['num_cpps'] = len(self.config['cpp_names'])
            self.optimizer = self.optimizer_class(param_bounds, ga_config)
    
    def _get_fallback_action(self, current_control_input):
        """Get safe fallback control action when optimizer fails.
        
        Implements multiple fallback strategies in order of preference:
        1. Last successful optimization result
        2. Hold current control input (if valid)
        3. Safe default control values
        
        Args:
            current_control_input: Current control input values
            
        Returns:
            np.ndarray: Safe control action
        """
        # Strategy 1: Use last successful optimization result
        if self._last_successful_action is not None:
            if self._validate_control_action(self._last_successful_action):
                return self._last_successful_action.copy()
        
        # Strategy 2: Hold current control input (if valid)
        if current_control_input is not None:
            if self._validate_control_action(current_control_input):
                return current_control_input.copy()
        
        # Strategy 3: Use pre-calculated safe default control values
        return self._calculate_safe_default_action()
    
    def _validate_control_action(self, action):
        """Validate that control action satisfies constraints.
        
        Args:
            action: Control action to validate
            
        Returns:
            bool: True if action is valid and safe
        """
        if action is None or not isinstance(action, np.ndarray):
            return False
            
        if action.size != len(self.config['cpp_names']):
            return False
            
        if not np.all(np.isfinite(action)):
            return False
            
        # Check constraint bounds
        cpp_config = self.config['cpp_constraints']
        for i, name in enumerate(self.config['cpp_names']):
            if name in cpp_config:
                min_val = cpp_config[name]['min_val']
                max_val = cpp_config[name]['max_val']
                if not (min_val <= action[i] <= max_val):
                    return False
                    
        return True

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
        
        # Validate soft sensor configuration for robust pharmaceutical control
        required_soft_sensor_base = ['spray_rate', 'carousel_speed']
        required_soft_sensors = ['specific_energy', 'froude_number_proxy']
        
        missing_soft_base = [var for var in required_soft_sensor_base if var not in self.config['cpp_full_names']]
        missing_soft_sensors = [var for var in required_soft_sensors if var not in self.config['cpp_full_names']]
        
        if missing_soft_base:
            raise ValueError(f"Missing required base variables for soft sensor calculations: {missing_soft_base}")
        if missing_soft_sensors:
            raise ValueError(f"Missing required soft sensor variables in cpp_full_names: {missing_soft_sensors}")
        
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
        
        # Validate optimizer reset configuration parameters
        if 'setpoint_change_threshold' in self.config:
            threshold = self.config['setpoint_change_threshold']
            if not isinstance(threshold, (int, float)) or threshold <= 0:
                raise ValueError("'setpoint_change_threshold' must be a positive number")
        
        if 'reset_optimizer_on_setpoint_change' in self.config:
            reset_flag = self.config['reset_optimizer_on_setpoint_change']
            if not isinstance(reset_flag, bool):
                raise ValueError("'reset_optimizer_on_setpoint_change' must be a boolean")
                
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
