import torch
import numpy as np
import pandas as pd
import itertools
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

class MPCController:
    """
    Implements a Model Predictive Controller that uses a trained PyTorch model
    to find optimal control actions while respecting process constraints.
    """
    def __init__(self, model, config, constraints, scalers):
        self.model = model
        self.config = config
        self.constraints = constraints
        self.scalers = scalers
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.model.eval()

    def _generate_control_lattice(self, current_cpps):
        """Creates a grid of possible future control sequences."""
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
        """Removes candidate sequences that violate process constraints."""
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

    def _calculate_cost(self, prediction, action, target_cmas, current_action):
        """Calculates the cost of a predicted trajectory."""
        # Ensure target is on the correct device
        target_cmas = target_cmas.to(self.device)
        current_action = current_action.to(self.device)

        # 1. Target Error Cost (how far are we from the setpoint?)
        # Using L1 loss (Mean Absolute Error) is often more robust to outliers
        target_error = torch.mean(torch.abs(prediction - target_cmas))

        # 2. Control Effort Cost (penalize large changes from current state)
        # FIXED: Compare first proposed action to current action, not to itself
        proposed_first_action = action[0, 0, :len(self.config['cpp_names'])]  # Only base CPPs, not soft sensors
        control_effort = torch.mean(torch.abs(proposed_first_action - current_action))

        # Combine costs with a weighting factor (lambda)
        total_cost = target_error + self.config['control_effort_lambda'] * control_effort
        return total_cost.item()

    def suggest_action(self, past_cmas_unscaled: pd.DataFrame, past_cpps_unscaled: pd.DataFrame, target_cmas_unscaled: np.ndarray) -> np.ndarray:
        """
        Find the optimal single control action using Model Predictive Control.
        
        Args:
            past_cmas_unscaled: DataFrame with unscaled CMA history, columns must match config['cma_names']
            past_cpps_unscaled: DataFrame with unscaled CPP history, columns must match config['cpp_names_and_soft_sensors'] 
            target_cmas_unscaled: Array of shape (horizon, num_cmas) with unscaled target values
            
        Returns:
            Array of shape (num_base_cpps,) with optimal unscaled CPP values
        """
        # Extract current CPPs directly from unscaled data
        current_cpps_unscaled = past_cpps_unscaled.iloc[-1][self.config['cpp_names']].values
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
        target_cmas_scaled = np.zeros_like(target_cmas_unscaled)
        for i, name in enumerate(self.config['cma_names']):
            target_df = pd.DataFrame(target_cmas_unscaled[:, i].reshape(-1, 1), columns=[name])
            target_cmas_scaled[:, i] = self.scalers[name].transform(target_df).flatten()
        target_cmas_tensor = torch.tensor(target_cmas_scaled, dtype=torch.float32).unsqueeze(0)

        # Scale the historical data for model input
        past_cmas_scaled = pd.DataFrame(index=past_cmas_unscaled.index)
        for col in self.config['cma_names']:
            past_cmas_scaled[col] = self.scalers[col].transform(past_cmas_unscaled[[col]]).flatten()
            
        past_cpps_scaled = pd.DataFrame(index=past_cpps_unscaled.index)
        for col in self.config['cpp_names_and_soft_sensors']:
            past_cpps_scaled[col] = self.scalers[col].transform(past_cpps_unscaled[[col]]).flatten()
        
        # Convert DataFrames to numpy arrays for tensor creation
        past_cmas_values = past_cmas_scaled.values
        past_cpps_values = past_cpps_scaled.values
            
        # Convert to tensors
        past_cmas_tensor = torch.tensor(past_cmas_values, dtype=torch.float32).unsqueeze(0).to(self.device)
        past_cpps_tensor = torch.tensor(past_cpps_values, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Create tensor for current CPPs (for control effort penalty)
        current_cpps_tensor = torch.tensor(current_cpps_unscaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            pbar = tqdm(valid_candidates_unscaled, desc="Evaluating MPC Candidates", leave=False)
            for action_seq_unscaled in pbar:
                # 1. Create the full 5-feature unscaled action sequence
                action_seq_with_sensors = np.zeros((action_seq_unscaled.shape[0], len(self.config['cpp_names_and_soft_sensors'])))
                action_seq_with_sensors[:, :action_seq_unscaled.shape[1]] = action_seq_unscaled
                
                # Calculate soft sensors
                spray_rate = action_seq_unscaled[:, 0]
                carousel_speed = action_seq_unscaled[:, 2]
                action_seq_with_sensors[:, 3] = (spray_rate * carousel_speed) / 1000.0  # specific_energy
                action_seq_with_sensors[:, 4] = (carousel_speed**2) / 9.81             # froude_number_proxy

                # 2. Scale the full 5-feature sequence
                action_df_unscaled = pd.DataFrame(action_seq_with_sensors, columns=self.config['cpp_names_and_soft_sensors'])
                action_df_scaled = pd.DataFrame(index=action_df_unscaled.index)
                for col in self.config['cpp_names_and_soft_sensors']:
                    action_df_scaled[col] = self.scalers[col].transform(action_df_unscaled[[col]])

                action_tensor = torch.tensor(action_df_scaled.values, dtype=torch.float32).unsqueeze(0).to(self.device)

                # Get model prediction
                prediction = self.model(past_cmas_tensor, past_cpps_tensor, action_tensor)

                # Calculate cost
                cost = self._calculate_cost(prediction, action_tensor, target_cmas_tensor, current_cpps_tensor)

                if cost < best_cost:
                    best_cost = cost
                    best_action_sequence = action_seq_unscaled

        # 4. Return the first step of the best plan found
        return best_action_sequence[0]
