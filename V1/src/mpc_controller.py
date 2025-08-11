import torch
import numpy as np
import itertools
from tqdm.auto import tqdm
import pandas as pd

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

    def _calculate_cost(self, prediction, action, target_cmas):
        """Calculates the cost of a predicted trajectory."""
        # Ensure target is on the correct device
        target_cmas = target_cmas.to(self.device)

        # 1. Target Error Cost (how far are we from the setpoint?)
        # Using L1 loss (Mean Absolute Error) is often more robust to outliers
        target_error = torch.mean(torch.abs(prediction - target_cmas))

        # 2. Control Effort Cost (penalize large changes to promote stability)
        # This is a placeholder; a more complex version could penalize deviation from a desired steady state
        control_effort = torch.mean(torch.abs(action - action[0])) # Penalize non-constant actions

        # Combine costs with a weighting factor (lambda)
        total_cost = target_error + self.config['control_effort_lambda'] * control_effort
        return total_cost.item()

    def suggest_action(self, past_cmas_scaled, past_cpps_scaled, target_cmas_unscaled):
        """The main MPC loop to find and return the best single control action."""
        # Get the last known CPPs (unscaled) to base our search on
        last_cpps_scaled = past_cpps_scaled.iloc[-1, :].values
        current_cpps_unscaled = np.zeros(len(self.config['cpp_names']))
        for i, name in enumerate(self.config['cpp_names']):
            current_cpps_unscaled[i] = self.scalers[name].inverse_transform(last_cpps_scaled[i].reshape(-1, 1)).flatten()[0]

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

        # Prepare scaled target tensor once
        target_cmas_scaled = np.zeros_like(target_cmas_unscaled)
        for i, name in enumerate(self.config['cma_names']):
            target_cmas_scaled[:, i] = self.scalers[name].transform(target_cmas_unscaled[:, i].reshape(-1, 1)).flatten()
        target_cmas_tensor = torch.tensor(target_cmas_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Convert historical data to tensors
        past_cmas_tensor = torch.tensor(past_cmas_scaled.values, dtype=torch.float32).unsqueeze(0).to(self.device)
        past_cpps_tensor = torch.tensor(past_cpps_scaled.values, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pbar = tqdm(valid_candidates_unscaled, desc="Evaluating MPC Candidates", leave=False)
            for action_seq_unscaled in pbar:
                # Scale the candidate action sequence for the model
                #action_seq_scaled = np.zeros_like(action_seq_unscaled)
                action_seq_scaled = np.zeros((action_seq_unscaled.shape[0], len(self.config['cpp_names_and_soft_sensors'])))
                # Add soft sensors to the action sequence
                action_seq_with_sensors = np.zeros((action_seq_unscaled.shape[0], len(self.config[ 'cpp_names_and_soft_sensors'])))
                action_seq_with_sensors[:, :action_seq_unscaled.shape[1]] = action_seq_unscaled
                spray_rate = action_seq_with_sensors[:, 0]
                carousel_speed = action_seq_with_sensors[:, 2]
                specific_energy = (spray_rate * carousel_speed) / 1000.0
                froude_number_proxy = (carousel_speed**2) / 9.81
                action_seq_with_sensors[:, 3] = specific_energy
                action_seq_with_sensors[:, 4] = froude_number_proxy
                for i, name in enumerate(self.config['cpp_names_and_soft_sensors']):
                    if name in self.scalers:
                         #action_seq_scaled[:, i] = self.scalers[name].transform(action_seq_with_sensors[:, i].reshape(-1, 1)).flatten()
                        # Pass as DataFrame to avoid UserWarning
                        action_seq_scaled[:, i] = self.scalers[name].transform(pd.DataFrame(action_seq_with_sensors[:, i],columns=[name])).flatten()

                action_tensor = torch.tensor(action_seq_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)

                # Get model prediction
                prediction = self.model(past_cmas_tensor, past_cpps_tensor, action_tensor)

                # Calculate cost
                cost = self._calculate_cost(prediction, action_tensor, target_cmas_tensor)

                if cost < best_cost:
                    best_cost = cost
                    best_action_sequence = action_seq_unscaled

        # 4. Return the first step of the best plan found
        return best_action_sequence[0]
