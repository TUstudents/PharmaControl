import torch
from torch.utils.data import Dataset
import pandas as pd

class GranulationDataset(Dataset):
    """
    Custom PyTorch Dataset for creating time-series sequences for the 
    granulation process predictive model.
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
        # The number of possible start points for a complete sequence
        return len(self.df) - self.lookback - self.horizon + 1

    def __getitem__(self, idx):
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
            torch.tensor(future_cmas_target, dtype=torch.float32)
        )
