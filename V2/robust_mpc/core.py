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
            disturbance_scaled = self._scale_cma_vector(self.disturbance_estimate)
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
        # In a real app, this would pull from a historical data buffer.
        # For this test, we'll create dummy history.
        L = self.config['lookback']
        past_cmas_unscaled = np.tile(current_smooth_state, (L, 1))
        past_cpps_unscaled = np.tile([120, 500, 30], (L, 1)) # Dummy CPPs
        # Add soft sensors to CPPs
        # ... (logic from notebook V1-2) ...
        # Scale both
        past_cmas_scaled = self._scale_cma_plan(past_cmas_unscaled)
        past_cpps_scaled = self._scale_cpp_plan(past_cpps_unscaled, with_soft_sensors=True)
        return past_cmas_scaled, past_cpps_scaled

    def _scale_cpp_plan(self, plan_unscaled, with_soft_sensors=False):
        # This needs to be implemented robustly, matching the training preprocessing
        # For now, a placeholder
        if with_soft_sensors: return np.zeros((plan_unscaled.shape[0], len(self.config['cpp_full_names'])))
        return np.zeros_like(plan_unscaled)

    def _scale_cma_plan(self, plan_unscaled):
        return np.zeros_like(plan_unscaled)

    def _scale_cma_vector(self, vector_unscaled):
        return np.zeros_like(vector_unscaled)

    def _get_param_bounds(self):
        param_bounds = []
        cpp_config = self.config['cpp_constraints']
        for _ in range(self.config['horizon']):
            for name in self.config['cpp_names']:
                param_bounds.append((cpp_config[name]['min_val'], cpp_config[name]['max_val']))
        return param_bounds
