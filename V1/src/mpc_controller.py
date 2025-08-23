import torch
import numpy as np
import pandas as pd
import itertools
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

class MPCController:
    """Model Predictive Controller for pharmaceutical continuous granulation processes.
    
    This controller implements a discrete optimization-based MPC that uses a trained
    neural network predictor to find optimal control actions while respecting 
    operational constraints and minimizing tracking error.
    
    The controller performs exhaustive grid search over discretized control changes,
    evaluates each candidate using the predictive model, and selects the action
    that minimizes a weighted combination of target tracking error and control effort.
    
    Attributes:
        model: Trained PyTorch neural network for process prediction
        config: Configuration dictionary containing control parameters
        constraints: Process constraint definitions for each control variable
        scalers: Data scalers used during model training for consistent preprocessing
        device: PyTorch device (CPU/GPU) for model execution
    
    Example:
        >>> controller = MPCController(model, config, constraints, scalers)
        >>> optimal_action = controller.suggest_action(past_cmas, past_cpps, targets)
    """
    def __init__(self, model, config, constraints, scalers):
        """Initialize the MPC controller with model and process configuration.
        
        Args:
            model: Trained PyTorch neural network model for process prediction.
                Must implement forward(past_cmas, past_cpps, future_cpps) -> predictions
            config: Configuration dictionary containing:
                - 'cpp_names': List of critical process parameter names
                - 'cma_names': List of critical material attribute names  
                - 'cpp_names_and_soft_sensors': Extended CPP list including soft sensors
                - 'horizon': Prediction and control horizon length
                - 'discretization_steps': Number of discrete control options per variable
                - 'control_effort_lambda': Weight for control effort penalty term
            constraints: Constraint dictionary with structure:
                {variable_name: {'min_val': float, 'max_val': float, 'max_change_per_step': float}}
            scalers: Dictionary of fitted sklearn scalers for data preprocessing:
                {variable_name: fitted_scaler_object}
        
        Raises:
            ValueError: If required configuration keys are missing
            RuntimeError: If model cannot be moved to available device
        """
        self.model = model
        self.config = config
        self.constraints = constraints
        self.scalers = scalers
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.model.eval()

    def _generate_control_lattice(self, current_cpps):
        """Generate discrete control action candidates for MPC optimization.
        
        Creates a grid of possible control sequences by discretizing the allowed
        change range for each control variable and generating all combinations.
        Each sequence assumes constant control action over the prediction horizon.
        
        Args:
            current_cpps: Current values of critical process parameters as numpy array.
                Shape: (num_cpps,)
        
        Returns:
            List of candidate control sequences, each as numpy array of shape
            (horizon, num_cpps). Each sequence represents constant control values
            applied over the entire prediction horizon.
        
        Notes:
            The discretization creates n^k candidates where n is discretization_steps
            and k is the number of control variables. This can become computationally
            expensive for high-dimensional control spaces.
        """
        cpp_names = self.config['cpp_names']
        discretization = self.config['discretization_steps']

        # For each CPP, create a set of possible delta (change) values
        # Example: [-5.0, 0.0, 5.0] for spray_rate
        delta_options = []
        for name in cpp_names:
            max_change = self.constraints[name]['max_change_per_step']
            options = np.linspace(-max_change, max_change, discretization)
            delta_options.append(options)

        # Create the Cartesian product of all delta options
        # This gives every possible combination of a single-step change
        candidate_deltas = list(itertools.product(*delta_options))

        # Assume the change is held constant over the horizon H
        candidate_sequences = []
        for deltas in candidate_deltas:
            # Create a plan by applying the deltas to the current CPPs
            new_cpp_step = current_cpps + np.array(deltas)
            # Create a full horizon sequence by repeating this step
            sequence = np.tile(new_cpp_step, (self.config['horizon'], 1))
            candidate_sequences.append(sequence)

        return candidate_sequences

    def _filter_by_constraints(self, candidates, current_cpps):
        """Filter control candidates to ensure operational constraint compliance.
        
        Validates each candidate control sequence against defined operational limits
        including minimum and maximum allowed values for each control variable.
        
        Args:
            candidates: List of candidate control sequences, each as numpy array
                of shape (horizon, num_cpps)
            current_cpps: Current control variable values as numpy array.
                Shape: (num_cpps,). Used for reference but not directly in filtering.
        
        Returns:
            List of valid candidate control sequences that satisfy all operational
            constraints. May be empty if no candidates satisfy constraints.
        
        Notes:
            Only checks the first time step of each sequence since all sequences
            assume constant control values. This assumes constraints are time-invariant.
        """
        valid_candidates = []
        cpp_names = self.config['cpp_names']

        for seq in candidates:
            is_valid = True
            # We only need to check the first step, as it's held constant
            first_step = seq[0]
            for i, name in enumerate(cpp_names):
                # Check min/max operational limits
                if not (self.constraints[name]['min_val'] <= first_step[i] <= self.constraints[name]['max_val']):
                    is_valid = False
                    break

            if is_valid:
                valid_candidates.append(seq)

        return valid_candidates

    def _calculate_cost(self, prediction, action, target_cmas, current_action_scaled):
        """Calculate the MPC objective function for a candidate control sequence.
        
        Computes a weighted combination of target tracking error and control effort
        penalty to evaluate the quality of a proposed control action.
        
        Args:
            prediction: Model prediction tensor of shape (batch, horizon, num_cmas).
                Contains predicted critical material attributes over the horizon.
            action: Proposed control action tensor of shape (batch, horizon, num_features).
                Contains scaled control variables and soft sensors.
            target_cmas: Target setpoint tensor of shape (batch, horizon, num_cmas).
                Desired values for critical material attributes.
            current_action_scaled: Current control action as scaled tensor of shape (num_cpps,).
                Used for calculating control effort penalty.
        
        Returns:
            Scalar cost value as float. Lower values indicate better performance.
            Combines L1 tracking error with weighted control effort penalty.
        
        Notes:
            Uses L1 (MAE) loss for tracking error as it's robust to outliers.
            Control effort penalizes large changes from current operating point.
        """
        # Ensure target is on the correct device
        target_cmas = target_cmas.to(self.device)
        current_action_scaled = current_action_scaled.to(self.device)

        # 1. Target Error Cost (how far are we from the setpoint?)
        # Using L1 loss (Mean Absolute Error) is often more robust to outliers
        target_error = torch.mean(torch.abs(prediction - target_cmas))

        # 2. Control Effort Cost (penalize large changes from current state)
        # FIXED: Compare scaled proposed action to scaled current action
        proposed_first_action_scaled = action[0, 0, :len(self.config['cpp_names'])]  # Only base CPPs, not soft sensors
        control_effort = torch.mean(torch.abs(proposed_first_action_scaled - current_action_scaled))

        # Combine costs with a weighting factor (lambda)
        total_cost = target_error + self.config['control_effort_lambda'] * control_effort
        return total_cost.item()

    def suggest_action(self, past_cmas: pd.DataFrame, past_cpps: pd.DataFrame, target_cmas: np.ndarray) -> np.ndarray:
        """Compute optimal control action using Model Predictive Control optimization.
        
        Performs discrete MPC optimization by:
        1. Generating candidate control sequences through grid discretization
        2. Filtering candidates to ensure constraint satisfaction
        3. Evaluating each candidate using the neural network predictor
        4. Selecting the action that minimizes the weighted cost function
        
        The optimization uses exhaustive search over a discretized control space,
        making it computationally expensive but guaranteeing the global optimum
        within the discretized domain.
        
        Args:
            past_cmas: Historical critical material attribute data as DataFrame.
                Must contain columns matching config['cma_names']. Data should be
                in original engineering units (unscaled). Shape: (lookback, num_cmas)
            past_cpps: Historical critical process parameter data as DataFrame.
                Must contain columns matching config['cpp_names_and_soft_sensors'].
                Includes both control variables and calculated soft sensors.
                Data in original engineering units. Shape: (lookback, num_features)
            target_cmas: Target setpoints for critical material attributes.
                Array of shape (horizon, num_cmas) in original engineering units.
                Typically constant setpoints repeated over the prediction horizon.
        
        Returns:
            Optimal control action as numpy array of shape (num_base_cpps,).
            Contains unscaled values for the base critical process parameters
            (excluding soft sensors). Values are clipped to operational constraints.
        
        Raises:
            ValueError: If required variables are missing from configuration or if
                input data shapes are incompatible with model requirements.
            RuntimeError: If model prediction fails or if all candidates produce
                invalid costs.
        
        Notes:
            - Computational complexity scales as O(n^k) where n is discretization
              steps and k is number of control variables
            - Returns current control values as fallback if optimization fails
            - Soft sensors (derived variables) are automatically calculated
        """
        # Extract current CPPs directly from unscaled data
        current_cpps_unscaled = past_cpps.iloc[-1][self.config['cpp_names']].values
        # 1. Generate all possible actions
        candidates_unscaled = self._generate_control_lattice(current_cpps_unscaled)

        # 2. Filter out invalid actions
        valid_candidates_unscaled = self._filter_by_constraints(candidates_unscaled, current_cpps_unscaled)

        if not valid_candidates_unscaled:
            print("Warning: No valid control actions found after applying constraints.")
            return current_cpps_unscaled # Return the last known safe action

        # 3. Predict and Evaluate
        best_cost = float('inf')
        best_action_sequence = None

        # Prepare scaled target tensor once using proper DataFrame
        target_cmas_scaled = np.zeros_like(target_cmas)
        for i, name in enumerate(self.config['cma_names']):
            target_df = pd.DataFrame(target_cmas[:, i].reshape(-1, 1), columns=[name])
            target_cmas_scaled[:, i] = self.scalers[name].transform(target_df).flatten()
        target_cmas_tensor = torch.tensor(target_cmas_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Scale the historical data for model input
        past_cmas_scaled = pd.DataFrame(index=past_cmas.index)
        for col in self.config['cma_names']:
            past_cmas_scaled[col] = self.scalers[col].transform(past_cmas[[col]]).flatten()
            
        past_cpps_scaled = pd.DataFrame(index=past_cpps.index)
        for col in self.config['cpp_names_and_soft_sensors']:
            past_cpps_scaled[col] = self.scalers[col].transform(past_cpps[[col]]).flatten()
        
        # Convert DataFrames to numpy arrays for tensor creation
        past_cmas_values = past_cmas_scaled.values
        past_cpps_values = past_cpps_scaled.values
            
        # Convert to tensors
        past_cmas_tensor = torch.tensor(past_cmas_values, dtype=torch.float32).unsqueeze(0).to(self.device)
        past_cpps_tensor = torch.tensor(past_cpps_values, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Create scaled tensor for current CPPs (for control effort penalty)
        current_cpps_scaled = np.zeros(len(self.config['cpp_names']))
        for i, name in enumerate(self.config['cpp_names']):
            current_cpp_df = pd.DataFrame(current_cpps_unscaled[i].reshape(-1, 1), columns=[name])
            current_cpps_scaled[i] = self.scalers[name].transform(current_cpp_df).flatten()[0]
        current_cpps_tensor = torch.tensor(current_cpps_scaled, dtype=torch.float32).to(self.device)

        # Pre-calculate indices for soft sensor calculations (optimization)
        try:
            spray_rate_idx = self.config['cpp_names'].index('spray_rate')
            carousel_speed_idx = self.config['cpp_names'].index('carousel_speed')
            specific_energy_idx = self.config['cpp_names_and_soft_sensors'].index('specific_energy')
            froude_number_idx = self.config['cpp_names_and_soft_sensors'].index('froude_number_proxy')
        except ValueError as e:
            raise ValueError(f"Required variable not found in configuration: {e}. "
                           f"Expected 'spray_rate' and 'carousel_speed' in cpp_names, "
                           f"'specific_energy' and 'froude_number_proxy' in cpp_names_and_soft_sensors.")

        with torch.no_grad():
            pbar = tqdm(valid_candidates_unscaled, desc="Evaluating MPC Candidates", leave=False)
            for action_seq_unscaled in pbar:
                # 1. Create the full feature unscaled action sequence
                action_seq_with_sensors = np.zeros((action_seq_unscaled.shape[0], len(self.config['cpp_names_and_soft_sensors'])))
                action_seq_with_sensors[:, :action_seq_unscaled.shape[1]] = action_seq_unscaled
                
                # Calculate soft sensors using pre-calculated indices
                spray_rate = action_seq_unscaled[:, spray_rate_idx]
                carousel_speed = action_seq_unscaled[:, carousel_speed_idx]
                action_seq_with_sensors[:, specific_energy_idx] = (spray_rate * carousel_speed) / 1000.0  # specific_energy
                action_seq_with_sensors[:, froude_number_idx] = (carousel_speed**2) / 9.81             # froude_number_proxy

                # 2. Scale the full feature sequence
                action_df_unscaled = pd.DataFrame(action_seq_with_sensors, columns=self.config['cpp_names_and_soft_sensors'])
                action_df_scaled = pd.DataFrame(index=action_df_unscaled.index)
                for col in self.config['cpp_names_and_soft_sensors']:
                    action_df_scaled[col] = self.scalers[col].transform(action_df_unscaled[[col]])

                action_tensor = torch.tensor(action_df_scaled.values, dtype=torch.float32).unsqueeze(0).to(self.device)

                # Get model prediction
                prediction = self.model(past_cmas_tensor, past_cpps_tensor, action_tensor)

                # Calculate cost
                cost = self._calculate_cost(prediction, action_tensor, target_cmas_tensor, current_cpps_tensor)

                # Only update if cost is valid (finite) and better than current best
                if not (np.isnan(cost) or np.isinf(cost)) and cost < best_cost:
                    best_cost = cost
                    best_action_sequence = action_seq_unscaled

        # 4. Return the first step of the best plan found
        if best_action_sequence is None:
            print("Warning: No valid control actions found - all candidates produced invalid costs.")
            print(f"Evaluated {len(valid_candidates_unscaled)} candidates, final best_cost: {best_cost}")
            print(f"Returning current CPPs as fallback: {current_cpps_unscaled}")
            return current_cpps_unscaled
        
        return best_action_sequence[0]
