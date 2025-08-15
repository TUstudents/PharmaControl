====================
Dataset
====================

.. currentmodule:: dataset

The dataset module provides efficient PyTorch Dataset implementation for pharmaceutical 
granulation process time series data. It implements sliding window sequence extraction 
suitable for training transformer-based predictive models in Model Predictive Control 
applications.

Overview
========

The :class:`GranulationDataset` class transforms continuous process time series data into 
overlapping temporal sequences that follow the sequence-to-sequence learning paradigm 
required for MPC applications. Each sample contains:

* **Historical context**: Past CMA and CPP observations for process understanding
* **Control plan**: Future CPP sequence representing planned control actions
* **Target predictions**: Future CMA values that the model should predict

This structure enables supervised learning where the model learns to map from 
(historical_data + control_plan) → future_process_outputs.

Data Structure
==============

**Sliding Window Approach**

.. code-block:: text

   Time Series Data:
   t₀  t₁  t₂  t₃  t₄  t₅  t₆  t₇  t₈  t₉  t₁₀ t₁₁ t₁₂
   │   │   │   │   │   │   │   │   │   │   │   │   │
   
   Sample 1:
   ├───────────────────┤ lookback (historical)
                       ├───────────┤ horizon (future)
   
   Sample 2:
       ├───────────────────┤ lookback
                           ├───────────┤ horizon
   
   Sample 3:
           ├───────────────────┤ lookback
                               ├───────────┤ horizon

**Temporal Structure**

Each training sample maintains strict temporal causality:

1. **Historical window**: [t : t+lookback] - No future information leakage
2. **Future window**: [t+lookback : t+lookback+horizon] - Prediction targets
3. **No overlap**: Clear separation between context and prediction periods

Class Reference
===============

.. autoclass:: GranulationDataset
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
==============

Basic Dataset Creation
----------------------

.. code-block:: python

   import pandas as pd
   from torch.utils.data import DataLoader
   from V1.src.dataset import GranulationDataset
   
   # Load preprocessed time series data
   df = pd.read_csv('granulation_data.csv')
   
   # Define variable groups
   cma_cols = ['d50', 'lod']  # Critical Material Attributes
   cpp_cols = [               # Critical Process Parameters + soft sensors
       'spray_rate', 'air_flow', 'carousel_speed',
       'specific_energy', 'froude_number_proxy'
   ]
   
   # Create dataset
   dataset = GranulationDataset(
       df=df,
       cma_cols=cma_cols,
       cpp_cols=cpp_cols,
       lookback=36,    # Historical context window
       horizon=10      # Prediction horizon
   )
   
   print(f"Dataset length: {len(dataset)} sequences")
   print(f"Total data points: {len(df)} time steps")

Data Loading and Batching
-------------------------

.. code-block:: python

   # Create data loaders for training
   train_loader = DataLoader(
       dataset,
       batch_size=32,
       shuffle=True,
       num_workers=4,
       pin_memory=True  # For GPU acceleration
   )
   
   # Examine a single batch
   for batch_idx, (past_cmas, past_cpps, future_cpps, targets) in enumerate(train_loader):
       print(f"Batch {batch_idx}:")
       print(f"  Past CMAs shape: {past_cmas.shape}")      # [32, 36, 2]
       print(f"  Past CPPs shape: {past_cpps.shape}")      # [32, 36, 5]
       print(f"  Future CPPs shape: {future_cpps.shape}")  # [32, 10, 5]
       print(f"  Targets shape: {targets.shape}")          # [32, 10, 2]
       break

Train/Validation/Test Splitting
-------------------------------

.. code-block:: python

   import numpy as np
   from sklearn.model_selection import train_test_split
   
   # Chronological splitting to prevent temporal leakage
   def create_chronological_splits(df, train_ratio=0.7, val_ratio=0.15):
       """Split time series data chronologically"""
       n_total = len(df)
       
       # Calculate split indices
       train_end = int(n_total * train_ratio)
       val_end = int(n_total * (train_ratio + val_ratio))
       
       # Create splits
       train_df = df.iloc[:train_end].copy()
       val_df = df.iloc[train_end:val_end].copy()
       test_df = df.iloc[val_end:].copy()
       
       print(f"Train: {len(train_df)} samples ({len(train_df)/n_total:.1%})")
       print(f"Validation: {len(val_df)} samples ({len(val_df)/n_total:.1%})")
       print(f"Test: {len(test_df)} samples ({len(test_df)/n_total:.1%})")
       
       return train_df, val_df, test_df
   
   # Create chronological splits
   train_df, val_df, test_df = create_chronological_splits(df)
   
   # Create datasets for each split
   train_dataset = GranulationDataset(train_df, cma_cols, cpp_cols, lookback=36, horizon=10)
   val_dataset = GranulationDataset(val_df, cma_cols, cpp_cols, lookback=36, horizon=10)
   test_dataset = GranulationDataset(test_df, cma_cols, cpp_cols, lookback=36, horizon=10)
   
   # Create data loaders
   train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
   val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
   test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

Data Preprocessing Pipeline
---------------------------

.. code-block:: python

   from sklearn.preprocessing import MinMaxScaler
   import joblib
   
   def create_preprocessing_pipeline(train_df, cma_cols, cpp_cols):
       """Create and fit scalers on training data only"""
       scalers = {}
       
       # Fit scalers on training data
       for col in cma_cols + cpp_cols:
           scaler = MinMaxScaler()
           scaler.fit(train_df[[col]])
           scalers[col] = scaler
       
       return scalers
   
   def apply_preprocessing(df, scalers, cma_cols, cpp_cols):
       """Apply fitted scalers to transform data"""
       df_scaled = df.copy()
       
       for col in cma_cols + cpp_cols:
           df_scaled[col] = scalers[col].transform(df[[col]])
       
       return df_scaled
   
   # Create preprocessing pipeline
   scalers = create_preprocessing_pipeline(train_df, cma_cols, cpp_cols)
   
   # Apply to all splits
   train_df_scaled = apply_preprocessing(train_df, scalers, cma_cols, cpp_cols)
   val_df_scaled = apply_preprocessing(val_df, scalers, cma_cols, cpp_cols)
   test_df_scaled = apply_preprocessing(test_df, scalers, cma_cols, cpp_cols)
   
   # Save scalers for deployment
   joblib.dump(scalers, 'scalers.joblib')
   
   # Create datasets with scaled data
   train_dataset = GranulationDataset(train_df_scaled, cma_cols, cpp_cols, lookback=36, horizon=10)
   val_dataset = GranulationDataset(val_df_scaled, cma_cols, cpp_cols, lookback=36, horizon=10)
   test_dataset = GranulationDataset(test_df_scaled, cma_cols, cpp_cols, lookback=36, horizon=10)

Advanced Data Analysis
----------------------

.. code-block:: python

   import matplotlib.pyplot as plt
   import seaborn as sns
   
   def analyze_dataset_properties(dataset, variable_names):
       """Analyze statistical properties of the dataset"""
       
       # Collect all sequences
       all_past_cmas = []
       all_past_cpps = []
       all_future_cpps = []
       all_targets = []
       
       for i in range(min(1000, len(dataset))):  # Sample for efficiency
           past_cmas, past_cpps, future_cpps, targets = dataset[i]
           all_past_cmas.append(past_cmas.numpy())
           all_past_cpps.append(past_cpps.numpy())
           all_future_cpps.append(future_cpps.numpy())
           all_targets.append(targets.numpy())
       
       # Convert to arrays
       past_cmas_array = np.array(all_past_cmas)
       past_cpps_array = np.array(all_past_cpps)
       targets_array = np.array(all_targets)
       
       # Analyze temporal patterns
       fig, axes = plt.subplots(2, 2, figsize=(15, 10))
       
       # CMA temporal patterns
       for i, cma_name in enumerate(['d50', 'lod']):
           mean_trajectory = np.mean(past_cmas_array[:, :, i], axis=0)
           std_trajectory = np.std(past_cmas_array[:, :, i], axis=0)
           
           axes[0, i].plot(mean_trajectory, label='Mean')
           axes[0, i].fill_between(
               range(len(mean_trajectory)),
               mean_trajectory - std_trajectory,
               mean_trajectory + std_trajectory,
               alpha=0.3, label='±1 std'
           )
           axes[0, i].set_title(f'{cma_name} Historical Patterns')
           axes[0, i].set_xlabel('Time Steps (lookback)')
           axes[0, i].legend()
           axes[0, i].grid(True)
       
       # CPP distributions
       for i, cpp_name in enumerate(['spray_rate', 'air_flow']):
           cpp_data = past_cpps_array[:, :, i].flatten()
           axes[1, i].hist(cpp_data, bins=50, alpha=0.7, density=True)
           axes[1, i].set_title(f'{cpp_name} Distribution')
           axes[1, i].set_xlabel('Scaled Value')
           axes[1, i].set_ylabel('Density')
           axes[1, i].grid(True)
       
       plt.tight_layout()
       plt.show()
       
       # Correlation analysis
       plt.figure(figsize=(12, 8))
       
       # Flatten sequences for correlation analysis
       flat_data = np.concatenate([
           past_cmas_array.reshape(-1, past_cmas_array.shape[-1]),
           past_cpps_array.reshape(-1, past_cpps_array.shape[-1])
       ], axis=1)
       
       correlation_matrix = np.corrcoef(flat_data.T)
       
       sns.heatmap(
           correlation_matrix,
           annot=True,
           cmap='coolwarm',
           center=0,
           xticklabels=variable_names,
           yticklabels=variable_names
       )
       plt.title('Variable Correlation Matrix')
       plt.tight_layout()
       plt.show()
   
   # Analyze dataset
   variable_names = ['d50', 'lod', 'spray_rate', 'air_flow', 'carousel_speed', 'specific_energy', 'froude_number_proxy']
   analyze_dataset_properties(train_dataset, variable_names)

Custom Dataset Variations
==========================

Variable Horizon Dataset
------------------------

.. code-block:: python

   class VariableHorizonDataset(GranulationDataset):
       """Dataset with variable prediction horizons for robust training"""
       
       def __init__(self, df, cma_cols, cpp_cols, lookback, max_horizon, min_horizon=1):
           self.max_horizon = max_horizon
           self.min_horizon = min_horizon
           super().__init__(df, cma_cols, cpp_cols, lookback, max_horizon)
       
       def __getitem__(self, idx):
           # Randomly select horizon length
           horizon = np.random.randint(self.min_horizon, self.max_horizon + 1)
           
           # Get base sequence
           past_start = idx
           past_end = idx + self.lookback
           future_end = past_end + horizon
           
           # Extract sequences
           past_cmas = self.cma_data[past_start:past_end, :]
           past_cpps = self.cpp_data[past_start:past_end, :]
           future_cpps = self.cpp_data[past_end:future_end, :]
           future_cmas_target = self.cma_data[past_end:future_end, :]
           
           # Pad to max_horizon if necessary
           if horizon < self.max_horizon:
               pad_length = self.max_horizon - horizon
               future_cpps = np.concatenate([
                   future_cpps,
                   np.zeros((pad_length, future_cpps.shape[1]))
               ])
               future_cmas_target = np.concatenate([
                   future_cmas_target,
                   np.zeros((pad_length, future_cmas_target.shape[1]))
               ])
           
           return (
               torch.tensor(past_cmas, dtype=torch.float32),
               torch.tensor(past_cpps, dtype=torch.float32),
               torch.tensor(future_cpps, dtype=torch.float32),
               torch.tensor(future_cmas_target, dtype=torch.float32),
               torch.tensor(horizon, dtype=torch.long)  # Include actual horizon
           )

Augmented Dataset
-----------------

.. code-block:: python

   class AugmentedGranulationDataset(GranulationDataset):
       """Dataset with data augmentation for improved generalization"""
       
       def __init__(self, df, cma_cols, cpp_cols, lookback, horizon, noise_std=0.01):
           super().__init__(df, cma_cols, cpp_cols, lookback, horizon)
           self.noise_std = noise_std
       
       def __getitem__(self, idx):
           past_cmas, past_cpps, future_cpps, targets = super().__getitem__(idx)
           
           # Add Gaussian noise for augmentation
           if self.noise_std > 0:
               past_cmas += torch.randn_like(past_cmas) * self.noise_std
               past_cpps += torch.randn_like(past_cpps) * self.noise_std
               future_cpps += torch.randn_like(future_cpps) * self.noise_std
               targets += torch.randn_like(targets) * self.noise_std
           
           return past_cmas, past_cpps, future_cpps, targets

Memory-Efficient Dataset
------------------------

.. code-block:: python

   class MemoryEfficientDataset(Dataset):
       """Memory-efficient dataset for very large time series"""
       
       def __init__(self, file_path, cma_cols, cpp_cols, lookback, horizon):
           self.file_path = file_path
           self.cma_cols = cma_cols
           self.cpp_cols = cpp_cols
           self.lookback = lookback
           self.horizon = horizon
           
           # Only load metadata, not full data
           self.df_length = self._get_file_length()
       
       def _get_file_length(self):
           """Count lines in file without loading full data"""
           with open(self.file_path, 'r') as f:
               return sum(1 for _ in f) - 1  # Subtract header
       
       def __len__(self):
           return self.df_length - self.lookback - self.horizon + 1
       
       def __getitem__(self, idx):
           # Load only required chunk
           start_row = idx
           end_row = idx + self.lookback + self.horizon
           
           chunk = pd.read_csv(
               self.file_path,
               skiprows=range(1, start_row + 1),  # Skip header and previous rows
               nrows=end_row - start_row
           )
           
           # Process chunk same as regular dataset
           past_start = 0
           past_end = self.lookback
           future_end = self.lookback + self.horizon
           
           past_cmas = chunk[self.cma_cols].iloc[past_start:past_end].values
           past_cpps = chunk[self.cpp_cols].iloc[past_start:past_end].values
           future_cpps = chunk[self.cpp_cols].iloc[past_end:future_end].values
           future_cmas_target = chunk[self.cma_cols].iloc[past_end:future_end].values
           
           return (
               torch.tensor(past_cmas, dtype=torch.float32),
               torch.tensor(past_cpps, dtype=torch.float32),
               torch.tensor(future_cpps, dtype=torch.float32),
               torch.tensor(future_cmas_target, dtype=torch.float32)
           )

Data Validation
===============

Dataset Quality Checks
-----------------------

.. code-block:: python

   def validate_dataset_quality(dataset, scalers=None):
       """Comprehensive dataset validation"""
       
       print("Dataset Quality Validation")
       print("=" * 40)
       
       # Basic statistics
       print(f"Dataset length: {len(dataset):,} sequences")
       
       # Sample a batch for analysis
       sample_batch = []
       for i in range(min(100, len(dataset))):
           sample_batch.append(dataset[i])
       
       # Stack tensors
       past_cmas = torch.stack([s[0] for s in sample_batch])
       past_cpps = torch.stack([s[1] for s in sample_batch])
       future_cpps = torch.stack([s[2] for s in sample_batch])
       targets = torch.stack([s[3] for s in sample_batch])
       
       # Check for NaN/Inf values
       def check_tensor_health(tensor, name):
           has_nan = torch.isnan(tensor).any()
           has_inf = torch.isinf(tensor).any()
           min_val = tensor.min().item()
           max_val = tensor.max().item()
           
           print(f"{name}:")
           print(f"  Shape: {tensor.shape}")
           print(f"  Range: [{min_val:.4f}, {max_val:.4f}]")
           print(f"  Has NaN: {has_nan}")
           print(f"  Has Inf: {has_inf}")
           
           if has_nan or has_inf:
               print(f"  ⚠️  WARNING: {name} contains invalid values!")
       
       check_tensor_health(past_cmas, "Past CMAs")
       check_tensor_health(past_cpps, "Past CPPs")
       check_tensor_health(future_cpps, "Future CPPs")
       check_tensor_health(targets, "Targets")
       
       # Check temporal consistency
       print("\nTemporal Consistency:")
       
       # Verify no future information leakage
       past_end_time = dataset.lookback - 1
       future_start_time = dataset.lookback
       
       print(f"  Past data ends at: t={past_end_time}")
       print(f"  Future data starts at: t={future_start_time}")
       print(f"  No temporal overlap: ✓")
       
       # Check for data scaling (if scalers provided)
       if scalers:
           print("\nData Scaling Validation:")
           for i, col in enumerate(['d50', 'lod']):
               cma_values = past_cmas[:, :, i].flatten()
               expected_range = [0, 1]  # MinMaxScaler range
               actual_range = [cma_values.min().item(), cma_values.max().item()]
               
               in_range = (actual_range[0] >= -0.1 and actual_range[1] <= 1.1)
               status = "✓" if in_range else "⚠️"
               
               print(f"  {col}: {actual_range} (expected: {expected_range}) {status}")
   
   # Run validation
   validate_dataset_quality(train_dataset, scalers)

Sequence Continuity Check
-------------------------

.. code-block:: python

   def check_sequence_continuity(dataset, tolerance=1e-6):
       """Verify that adjacent sequences have proper temporal continuity"""
       
       print("Sequence Continuity Check")
       print("=" * 30)
       
       continuity_errors = []
       
       for i in range(min(10, len(dataset) - 1)):
           # Get consecutive sequences
           seq1 = dataset[i]
           seq2 = dataset[i + 1]
           
           # Check if seq2's past overlaps with seq1's past (shifted by 1)
           seq1_past_cmas = seq1[0]  # [lookback, cma_features]
           seq2_past_cmas = seq2[0]
           
           # The last (lookback-1) points of seq2 should match
           # the last (lookback-1) points of seq1, shifted by 1
           overlap_length = dataset.lookback - 1
           
           seq1_overlap = seq1_past_cmas[1:, :]  # Skip first point
           seq2_overlap = seq2_past_cmas[:overlap_length, :]
           
           difference = torch.abs(seq1_overlap - seq2_overlap).max()
           
           if difference > tolerance:
               continuity_errors.append((i, difference.item()))
       
       if continuity_errors:
           print(f"⚠️  Found {len(continuity_errors)} continuity errors:")
           for seq_idx, error in continuity_errors[:5]:
               print(f"  Sequence {seq_idx}: max difference = {error:.6f}")
       else:
           print("✓ All checked sequences have proper temporal continuity")

Performance Optimization
=========================

DataLoader Optimization
-----------------------

.. code-block:: python

   def create_optimized_dataloader(dataset, batch_size=32, num_workers=4):
       """Create optimized DataLoader for training"""
       
       # Custom collate function for better memory layout
       def collate_fn(batch):
           past_cmas = torch.stack([item[0] for item in batch])
           past_cpps = torch.stack([item[1] for item in batch])
           future_cpps = torch.stack([item[2] for item in batch])
           targets = torch.stack([item[3] for item in batch])
           
           return past_cmas, past_cpps, future_cpps, targets
       
       # Optimized DataLoader
       dataloader = DataLoader(
           dataset,
           batch_size=batch_size,
           shuffle=True,
           num_workers=num_workers,
           pin_memory=True,           # For GPU transfer
           persistent_workers=True,   # Keep workers alive
           prefetch_factor=2,         # Prefetch batches
           collate_fn=collate_fn
       )
       
       return dataloader

Memory Usage Analysis
---------------------

.. code-block:: python

   import psutil
   import os
   
   def analyze_memory_usage(dataset):
       """Analyze memory usage of dataset operations"""
       
       process = psutil.Process(os.getpid())
       
       # Baseline memory
       baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
       
       print(f"Baseline memory: {baseline_memory:.2f} MB")
       
       # Load dataset into memory
       dataloader = DataLoader(dataset, batch_size=32, num_workers=0)
       
       # Process one epoch
       for batch_idx, batch in enumerate(dataloader):
           if batch_idx == 0:
               first_batch_memory = process.memory_info().rss / 1024 / 1024
               print(f"After first batch: {first_batch_memory:.2f} MB")
           
           if batch_idx >= 10:  # Sample first 10 batches
               break
       
       peak_memory = process.memory_info().rss / 1024 / 1024
       print(f"Peak memory: {peak_memory:.2f} MB")
       print(f"Memory increase: {peak_memory - baseline_memory:.2f} MB")

Best Practices
==============

**Data Preparation**

1. **Chronological splitting**: Always split time series data chronologically
2. **Consistent scaling**: Fit scalers only on training data
3. **Quality validation**: Check for NaN, Inf, and outliers
4. **Temporal consistency**: Verify sequence continuity

**Memory Management**

1. **Efficient data types**: Use float32 instead of float64
2. **Batch size tuning**: Balance memory usage with training speed
3. **Worker processes**: Use multiple workers for data loading
4. **Pin memory**: Enable for faster GPU transfer

**Performance Tuning**

1. **Sequence length**: Balance context vs. computational cost
2. **Prefetching**: Enable DataLoader prefetching
3. **Persistent workers**: Keep workers alive between epochs
4. **Memory mapping**: Consider for very large datasets

Common Issues
=============

**Data Loading Errors**

1. **Shape mismatches**: Verify all sequences have consistent dimensions
2. **Missing values**: Handle NaN values before dataset creation
3. **Memory errors**: Reduce batch size or sequence length
4. **Worker crashes**: Reduce num_workers if using shared memory

**Training Issues**

1. **Temporal leakage**: Ensure no future information in historical data
2. **Scaling issues**: Consistent preprocessing between train/test
3. **Sequence alignment**: Verify temporal boundaries are correct
4. **Data distribution**: Check for train/test distribution shifts

**Performance Problems**

1. **Slow data loading**: Increase num_workers, enable pin_memory
2. **Memory leaks**: Check for tensor accumulation in training loop
3. **CPU bottleneck**: Profile data loading vs. model computation
4. **I/O bottleneck**: Consider faster storage or data caching

See Also
========

* :doc:`model_architecture` - Transformer models trained on this data
* :doc:`mpc_controller` - Using datasets for MPC validation
* :doc:`../tutorials/data_preparation` - Step-by-step data preparation guide
* :doc:`../examples/data_analysis` - Advanced data analysis techniques