import torch
from torch.utils.data import Dataset
import pandas as pd


class GranulationDataset(Dataset):
    """PyTorch Dataset for pharmaceutical granulation process time series modeling.

    This dataset class implements efficient sliding window sequence extraction from
    continuous granulation process data for training transformer-based predictive models.
    It creates overlapping temporal sequences suitable for supervised learning of
    multi-step-ahead process predictions.

    The dataset supports the sequence-to-sequence learning paradigm required for
    Model Predictive Control applications, where the model learns to predict future
    Critical Material Attributes (CMAs) based on historical process data and
    planned future control actions (CPPs).

    Data Structure:
    - Historical Context: (lookback, features) window of past CMAs and CPPs
    - Control Plan: (horizon, features) sequence of future planned CPPs
    - Target Predictions: (horizon, features) ground truth future CMAs to predict

    The sliding window approach maximizes data utilization from continuous process
    datasets while maintaining temporal causality for realistic MPC training.

    Args:
        df: Pandas DataFrame containing complete process time series data.
            Must include all columns specified in cma_cols and cpp_cols.
            Data should be chronologically ordered and preprocessed (scaled).
        cma_cols: List of column names for Critical Material Attributes (outputs).
            Typically includes variables like 'd50' (particle size), 'lod' (moisture).
        cpp_cols: List of column names for Critical Process Parameters (inputs).
            Includes control variables and soft sensors like 'spray_rate', 'air_flow',
            'carousel_speed', 'specific_energy', 'froude_number_proxy'.
        lookback: Number of historical time steps to include in model input.
            Determines the temporal context window for prediction. Typical range: 20-50.
        horizon: Number of future time steps to predict.
            Defines the prediction horizon for MPC applications. Typical range: 5-20.

    Attributes:
        df: Original DataFrame reference (kept for metadata access)
        cma_cols: CMA column names for output variable identification
        cpp_cols: CPP column names for input variable identification
        lookback: Historical window length parameter
        horizon: Prediction horizon length parameter
        cma_data: Numpy array of CMA data for efficient indexing
        cpp_data: Numpy array of CPP data for efficient indexing

    Returns:
        Each __getitem__ call returns a 4-tuple of torch.float32 tensors:
        - past_cmas: (lookback, num_cmas) historical material attributes
        - past_cpps: (lookback, num_cpps) historical process parameters
        - future_cpps: (horizon, num_cpps) planned future control actions
        - future_cmas_target: (horizon, num_cmas) ground truth targets for training

    Example:
        >>> df = pd.read_csv('granulation_data.csv')  # Preprocessed time series
        >>> dataset = GranulationDataset(
        ...     df=df,
        ...     cma_cols=['d50', 'lod'],
        ...     cpp_cols=['spray_rate', 'air_flow', 'carousel_speed', 'specific_energy', 'froude_number_proxy'],
        ...     lookback=36,
        ...     horizon=10
        ... )
        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> for batch in dataloader:
        ...     past_cmas, past_cpps, future_cpps, targets = batch
        ...     predictions = model(past_cmas, past_cpps, future_cpps)
        ...     loss = criterion(predictions, targets)

    Notes:
        - Data conversion to numpy arrays improves indexing performance
        - Temporal sequences maintain strict chronological order
        - Dataset length accounts for required lookback and horizon windows
        - All tensors are returned as float32 for GPU compatibility
    """

    def __init__(self, df, cma_cols, cpp_cols, lookback, horizon):
        self.df = df
        self.cma_cols = cma_cols
        self.cpp_cols = cpp_cols
        self.lookback = lookback
        self.horizon = horizon

        # Convert to numpy for faster slicing
        self.cma_data = df[cma_cols].to_numpy()
        self.cpp_data = df[cpp_cols].to_numpy()

    def __len__(self):
        """Return the total number of valid sequences in the dataset.

        Calculates the maximum number of complete sequences that can be extracted
        from the time series data given the lookback and horizon requirements.
        Each sequence needs sufficient historical data (lookback) and future data
        (horizon) to be valid for training.

        Returns:
            Integer count of valid sequences. Formula: total_rows - lookback - horizon + 1

        Notes:
            - Accounts for the fact that sequences near the end cannot provide full horizon
            - Sequences near the beginning lack sufficient historical context
            - Used by PyTorch DataLoader for batching and iteration control
        """
        # The number of possible start points for a complete sequence
        return len(self.df) - self.lookback - self.horizon + 1

    def __getitem__(self, idx):
        """Extract a single training sequence from the dataset.

        Implements the core sliding window logic to create a complete training sample
        with historical context, control plan, and prediction targets. This method
        defines the temporal structure that the transformer model will learn to map
        from inputs to outputs.

        Sequence Structure:
        - Historical window: [idx : idx + lookback] for context building
        - Future window: [idx + lookback : idx + lookback + horizon] for predictions
        - No temporal overlap between historical context and prediction targets
        - Maintains strict chronological ordering within each sequence

        Args:
            idx: Integer index specifying the starting position of the sequence
                in the overall time series. Must be in range [0, len(dataset)-1].

        Returns:
            Tuple of four torch.float32 tensors representing a complete training sample:

            past_cmas: Historical CMA observations, shape (lookback, num_cmas)
                Contains the process output measurements that provide context for prediction.
                Represents "what we observed" in terms of material quality attributes.

            past_cpps: Historical CPP measurements, shape (lookback, num_cpps)
                Contains the control input and soft sensor values that influenced past CMAs.
                Represents "what we did" in terms of process control actions.

            future_cpps: Planned control sequence, shape (horizon, num_cpps)
                Contains the control actions for which predictions are desired.
                Represents "what we plan to do" - the MPC control sequence to evaluate.

            future_cmas_target: Ground truth CMA values, shape (horizon, num_cmas)
                Contains the actual measured CMA values that the model should predict.
                Used as supervised learning targets during training.

        Notes:
            - All array slicing uses numpy for performance (converted to tensors at end)
            - Temporal boundaries are carefully managed to prevent data leakage
            - Float32 tensors ensure compatibility with most PyTorch models and GPUs
            - Sequence extraction maintains the causal structure required for MPC training
        """
        # Define the slice boundaries for the sample
        past_start = idx
        past_end = idx + self.lookback
        future_end = past_end + self.horizon

        # --- Extract sequences ---
        # Historical CMAs (what we observed)
        past_cmas = self.cma_data[past_start:past_end, :]

        # Historical CPPs (what we did)
        past_cpps = self.cpp_data[past_start:past_end, :]

        # Future CPPs (what we plan to do)
        future_cpps = self.cpp_data[past_end:future_end, :]

        # Future CMAs (the ground truth we want to predict)
        future_cmas_target = self.cma_data[past_end:future_end, :]

        # Convert to PyTorch tensors
        return (
            torch.tensor(past_cmas, dtype=torch.float32),
            torch.tensor(past_cpps, dtype=torch.float32),
            torch.tensor(future_cpps, dtype=torch.float32),
            torch.tensor(future_cmas_target, dtype=torch.float32),
        )
