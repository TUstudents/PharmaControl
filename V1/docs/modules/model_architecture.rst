====================
Model Architecture
====================

.. currentmodule:: model_architecture

The model architecture module implements a transformer-based encoder-decoder neural network 
specifically designed for pharmaceutical granulation process prediction. This architecture 
enables multi-step-ahead forecasting necessary for Model Predictive Control applications.

Overview
========

The module provides two main components:

* :class:`PositionalEncoding` - Sinusoidal position embeddings for sequence modeling
* :class:`GranulationPredictor` - Complete transformer architecture for process prediction

The transformer architecture is specifically adapted for process control applications, where 
the model must predict future Critical Material Attributes (CMAs) based on historical 
process data and planned future control actions (CPPs).

Transformer Architecture
========================

The :class:`GranulationPredictor` implements a sequence-to-sequence transformer with the 
following structure:

.. code-block:: text

   Historical Data    Future Controls    Predictions
   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
   │ Past CMAs   │    │ Future CPPs │    │ Future CMAs │
   │ Past CPPs   │    │             │    │             │
   └─────────────┘    └─────────────┘    └─────────────┘
          │                   │                   │
          ▼                   ▼                   ▼
   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
   │ Input Embed │    │ Input Embed │    │ Output Pred │
   └─────────────┘    └─────────────┘    └─────────────┘
          │                   │                   ▲
          ▼                   ▼                   │
   ┌─────────────┐    ┌─────────────┐             │
   │ Pos Encode  │    │ Pos Encode  │             │
   └─────────────┘    └─────────────┘             │
          │                   │                   │
          ▼                   ▼                   │
   ┌─────────────┐    ┌─────────────┐             │
   │ Transformer │    │ Transformer │             │
   │ Encoder     │────│ Decoder     │─────────────┘
   └─────────────┘    └─────────────┘

Mathematical Foundation
=======================

**Attention Mechanism**

The transformer uses scaled dot-product attention:

.. math::

   \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V

Where Q, K, V are query, key, and value matrices derived from input embeddings.

**Positional Encoding**

Sinusoidal positional encodings inject temporal information:

.. math::

   PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)

.. math::

   PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)

Where:
* pos = position in sequence
* i = dimension index  
* d_model = model dimensionality

**Process-Specific Adaptations**

For granulation process modeling:

* **Input sequences**: Historical CMAs + CPPs (context)
* **Target sequences**: Future CPPs (control plan)  
* **Output sequences**: Future CMAs (predictions)
* **Causal masking**: Prevents decoder from seeing future control actions

Class Reference
===============

.. autoclass:: PositionalEncoding
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: GranulationPredictor
   :members:
   :undoc-members:
   :show-inheritance:

Architecture Details
====================

**Model Dimensions**

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Default
     - Description
   * - d_model
     - 64
     - Hidden dimension for embeddings and attention
   * - nhead
     - 4
     - Number of attention heads (d_model must be divisible)
   * - num_encoder_layers
     - 2
     - Depth of encoder stack
   * - num_decoder_layers
     - 2
     - Depth of decoder stack
   * - dim_feedforward
     - 256
     - Hidden dimension in feedforward networks
   * - dropout
     - 0.1
     - Dropout probability for regularization

**Layer Structure**

1. **Input Embeddings**: Linear projections mapping features to d_model
2. **Positional Encoding**: Sinusoidal position information
3. **Encoder Stack**: Self-attention layers processing historical context
4. **Decoder Stack**: Cross-attention layers generating predictions
5. **Output Projection**: Linear layer mapping to CMA feature space

**Parameter Count**

For typical configuration (d_model=64, 2 layers, 4 heads):

.. math::

   \text{Parameters} \approx 4 \times d_{model}^2 \times \text{layers} + \text{embeddings}

Approximately 50,000-100,000 parameters for pharmaceutical applications.

Usage Examples
==============

Basic Model Creation
--------------------

.. code-block:: python

   import torch
   from V1.src.model_architecture import GranulationPredictor
   
   # Define feature dimensions
   cma_features = 2    # d50, lod
   cpp_features = 5    # spray_rate, air_flow, carousel_speed, specific_energy, froude_number_proxy
   
   # Create model with default architecture
   model = GranulationPredictor(
       cma_features=cma_features,
       cpp_features=cpp_features
   )
   
   print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

Custom Architecture
-------------------

.. code-block:: python

   # Larger model for complex processes
   model = GranulationPredictor(
       cma_features=2,
       cpp_features=5,
       d_model=128,           # Larger hidden dimension
       nhead=8,               # More attention heads
       num_encoder_layers=4,  # Deeper encoder
       num_decoder_layers=4,  # Deeper decoder
       dim_feedforward=512,   # Larger feedforward
       dropout=0.15           # Higher dropout for regularization
   )

Forward Pass Example
--------------------

.. code-block:: python

   # Create sample data
   batch_size = 32
   lookback = 36    # Historical window
   horizon = 10     # Prediction horizon
   
   # Historical data (encoder input)
   past_cmas = torch.randn(batch_size, lookback, cma_features)
   past_cpps = torch.randn(batch_size, lookback, cpp_features)
   
   # Future control plan (decoder input)
   future_cpps = torch.randn(batch_size, horizon, cpp_features)
   
   # Forward pass
   model.eval()
   with torch.no_grad():
       predictions = model(past_cmas, past_cpps, future_cpps)
   
   print(f"Input shapes:")
   print(f"  Past CMAs: {past_cmas.shape}")
   print(f"  Past CPPs: {past_cpps.shape}")
   print(f"  Future CPPs: {future_cpps.shape}")
   print(f"Output shape: {predictions.shape}")

Training Setup
--------------

.. code-block:: python

   import torch.nn as nn
   import torch.optim as optim
   from torch.utils.data import DataLoader
   from V1.src.dataset import GranulationDataset
   
   # Model setup
   model = GranulationPredictor(cma_features=2, cpp_features=5)
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model.to(device)
   
   # Training configuration
   criterion = nn.MSELoss()
   optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
   scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
   
   # Dataset and dataloader
   train_dataset = GranulationDataset(
       df=train_df,
       cma_cols=['d50', 'lod'],
       cpp_cols=['spray_rate', 'air_flow', 'carousel_speed', 'specific_energy', 'froude_number_proxy'],
       lookback=36,
       horizon=10
   )
   
   train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
   
   # Training loop
   model.train()
   for epoch in range(100):
       total_loss = 0.0
       
       for batch_idx, (past_cmas, past_cpps, future_cpps, targets) in enumerate(train_loader):
           # Move to device
           past_cmas = past_cmas.to(device)
           past_cpps = past_cpps.to(device)
           future_cpps = future_cpps.to(device)
           targets = targets.to(device)
           
           # Forward pass
           optimizer.zero_grad()
           predictions = model(past_cmas, past_cpps, future_cpps)
           loss = criterion(predictions, targets)
           
           # Backward pass
           loss.backward()
           torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
           optimizer.step()
           
           total_loss += loss.item()
       
       avg_loss = total_loss / len(train_loader)
       scheduler.step(avg_loss)
       
       if epoch % 10 == 0:
           print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")

Model Analysis
--------------

.. code-block:: python

   # Analyze model complexity
   def count_parameters(model):
       return sum(p.numel() for p in model.parameters() if p.requires_grad)
   
   def analyze_model_architecture(model):
       print("Model Architecture Analysis")
       print("=" * 40)
       print(f"Total Parameters: {count_parameters(model):,}")
       
       # Analyze each component
       encoder_params = sum(p.numel() for p in model.transformer.encoder.parameters())
       decoder_params = sum(p.numel() for p in model.transformer.decoder.parameters())
       embedding_params = (sum(p.numel() for p in model.cma_encoder_embedding.parameters()) +
                          sum(p.numel() for p in model.cpp_encoder_embedding.parameters()) +
                          sum(p.numel() for p in model.cpp_decoder_embedding.parameters()))
       output_params = sum(p.numel() for p in model.output_linear.parameters())
       
       print(f"Encoder Parameters: {encoder_params:,}")
       print(f"Decoder Parameters: {decoder_params:,}")
       print(f"Embedding Parameters: {embedding_params:,}")
       print(f"Output Parameters: {output_params:,}")
       
       # Memory estimation (rough)
       param_memory = count_parameters(model) * 4 / (1024**2)  # 4 bytes per float32
       print(f"Parameter Memory: {param_memory:.2f} MB")
   
   analyze_model_architecture(model)

Attention Visualization
-----------------------

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np
   
   def visualize_attention_weights(model, past_cmas, past_cpps, future_cpps):
       """Extract and visualize attention weights from transformer"""
       model.eval()
       
       # Hook to capture attention weights
       attention_weights = {}
       
       def save_attention(name):
           def hook(module, input, output):
               if hasattr(output, 'attn_weights'):
                   attention_weights[name] = output.attn_weights.detach()
           return hook
       
       # Register hooks (this is simplified - actual implementation depends on PyTorch version)
       hooks = []
       for name, module in model.transformer.named_modules():
           if 'attention' in name.lower():
               hooks.append(module.register_forward_hook(save_attention(name)))
       
       try:
           # Forward pass
           with torch.no_grad():
               predictions = model(past_cmas, past_cpps, future_cpps)
           
           # Plot attention patterns
           if attention_weights:
               fig, axes = plt.subplots(1, len(attention_weights), figsize=(15, 4))
               
               for idx, (name, weights) in enumerate(attention_weights.items()):
                   # Average over batch and heads
                   avg_weights = weights[0].mean(dim=0).cpu().numpy()
                   
                   im = axes[idx].imshow(avg_weights, cmap='Blues', aspect='auto')
                   axes[idx].set_title(f'Attention: {name}')
                   axes[idx].set_xlabel('Key Position')
                   axes[idx].set_ylabel('Query Position')
                   plt.colorbar(im, ax=axes[idx])
               
               plt.tight_layout()
               plt.show()
               
       finally:
           # Remove hooks
           for hook in hooks:
               hook.remove()
   
   # Example usage
   visualize_attention_weights(model, past_cmas[:1], past_cpps[:1], future_cpps[:1])

Model Interpretation
====================

**Feature Importance Analysis**

.. code-block:: python

   def analyze_feature_importance(model, test_loader, device):
       """Analyze which input features are most important for predictions"""
       model.eval()
       
       feature_importance = {
           'cma_features': ['d50', 'lod'],
           'cpp_features': ['spray_rate', 'air_flow', 'carousel_speed', 'specific_energy', 'froude_number_proxy']
       }
       
       baseline_loss = 0.0
       feature_losses = {}
       
       criterion = nn.MSELoss()
       
       # Calculate baseline loss
       with torch.no_grad():
           for past_cmas, past_cpps, future_cpps, targets in test_loader:
               past_cmas, past_cpps, future_cpps, targets = [x.to(device) for x in [past_cmas, past_cpps, future_cpps, targets]]
               predictions = model(past_cmas, past_cpps, future_cpps)
               baseline_loss += criterion(predictions, targets).item()
       
       baseline_loss /= len(test_loader)
       
       # Test importance of each feature by zeroing it out
       for feature_type in ['cma', 'cpp']:
           features = feature_importance[f'{feature_type}_features']
           
           for i, feature_name in enumerate(features):
               total_loss = 0.0
               
               with torch.no_grad():
                   for past_cmas, past_cpps, future_cpps, targets in test_loader:
                       past_cmas, past_cpps, future_cpps, targets = [x.to(device) for x in [past_cmas, past_cpps, future_cpps, targets]]
                       
                       # Zero out specific feature
                       if feature_type == 'cma':
                           past_cmas_modified = past_cmas.clone()
                           past_cmas_modified[:, :, i] = 0
                           predictions = model(past_cmas_modified, past_cpps, future_cpps)
                       else:
                           past_cpps_modified = past_cpps.clone()
                           future_cpps_modified = future_cpps.clone()
                           past_cpps_modified[:, :, i] = 0
                           future_cpps_modified[:, :, i] = 0
                           predictions = model(past_cmas, past_cpps_modified, future_cpps_modified)
                       
                       total_loss += criterion(predictions, targets).item()
               
               avg_loss = total_loss / len(test_loader)
               importance = (avg_loss - baseline_loss) / baseline_loss
               feature_losses[feature_name] = importance
       
       # Plot feature importance
       features = list(feature_losses.keys())
       importances = list(feature_losses.values())
       
       plt.figure(figsize=(10, 6))
       bars = plt.bar(features, importances)
       plt.title('Feature Importance Analysis')
       plt.xlabel('Features')
       plt.ylabel('Relative Importance (% increase in loss)')
       plt.xticks(rotation=45)
       
       # Color bars by importance
       for bar, importance in zip(bars, importances):
           if importance > 0.1:
               bar.set_color('red')
           elif importance > 0.05:
               bar.set_color('orange')
           else:
               bar.set_color('green')
       
       plt.tight_layout()
       plt.show()
       
       return feature_losses

**Prediction Quality Analysis**

.. code-block:: python

   def analyze_prediction_quality(model, test_loader, device, scalers):
       """Analyze prediction quality across different time horizons"""
       model.eval()
       
       horizon_errors = {i: [] for i in range(10)}  # Assuming 10-step horizon
       
       with torch.no_grad():
           for past_cmas, past_cpps, future_cpps, targets in test_loader:
               past_cmas, past_cpps, future_cpps, targets = [x.to(device) for x in [past_cmas, past_cpps, future_cpps, targets]]
               
               predictions = model(past_cmas, past_cpps, future_cpps)
               
               # Calculate errors for each time step
               for t in range(predictions.shape[1]):
                   step_error = torch.mean(torch.abs(predictions[:, t, :] - targets[:, t, :]), dim=0)
                   horizon_errors[t].extend(step_error.cpu().numpy())
       
       # Plot prediction quality vs. horizon
       horizons = list(range(10))
       d50_errors = [np.mean([errors[0] for errors in horizon_errors[h]]) for h in horizons]
       lod_errors = [np.mean([errors[1] for errors in horizon_errors[h]]) for h in horizons]
       
       fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
       
       ax1.plot(horizons, d50_errors, 'b-o')
       ax1.set_title('d50 Prediction Error vs. Horizon')
       ax1.set_xlabel('Prediction Step')
       ax1.set_ylabel('MAE (scaled units)')
       ax1.grid(True)
       
       ax2.plot(horizons, lod_errors, 'r-o')
       ax2.set_title('LOD Prediction Error vs. Horizon')
       ax2.set_xlabel('Prediction Step')
       ax2.set_ylabel('MAE (scaled units)')
       ax2.grid(True)
       
       plt.tight_layout()
       plt.show()

Performance Optimization
=========================

**Memory Optimization**

.. code-block:: python

   # Gradient checkpointing for large models
   from torch.utils.checkpoint import checkpoint
   
   class MemoryEfficientGranulationPredictor(GranulationPredictor):
       def forward(self, past_cmas, past_cpps, future_cpps):
           # Use gradient checkpointing for transformer layers
           past_cma_emb = self.cma_encoder_embedding(past_cmas)
           past_cpp_emb = self.cpp_encoder_embedding(past_cpps)
           src = self.pos_encoder(past_cma_emb + past_cpp_emb)
           
           tgt = self.pos_encoder(self.cpp_decoder_embedding(future_cpps))
           tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
           
           # Checkpoint transformer forward pass
           output = checkpoint(self.transformer, src, tgt, tgt_mask)
           return self.output_linear(output)

**Inference Optimization**

.. code-block:: python

   # TorchScript compilation for faster inference
   def optimize_for_inference(model, example_inputs):
       model.eval()
       
       # Trace the model
       traced_model = torch.jit.trace(model, example_inputs)
       
       # Optimize for inference
       traced_model = torch.jit.optimize_for_inference(traced_model)
       
       return traced_model
   
   # Example usage
   example_past_cmas = torch.randn(1, 36, 2)
   example_past_cpps = torch.randn(1, 36, 5)
   example_future_cpps = torch.randn(1, 10, 5)
   
   optimized_model = optimize_for_inference(
       model, 
       (example_past_cmas, example_past_cpps, example_future_cpps)
   )

Design Considerations
=====================

**Sequence Length Constraints**

* **Maximum length**: Limited by positional encoding (default: 5000)
* **Memory scaling**: O(L²) attention complexity where L = sequence length
* **Practical limits**: 50-100 steps for real-time applications

**Feature Engineering**

* **Soft sensors**: Include physics-based derived features
* **Normalization**: Essential for stable training and attention
* **Temporal alignment**: Ensure consistent sampling rates across sensors

**Architecture Choices**

* **Encoder depth**: 2-4 layers typical for process data
* **Attention heads**: 4-8 heads balance expressiveness vs. computation
* **Hidden dimension**: 64-128 sufficient for pharmaceutical processes
* **Dropout**: 0.1-0.2 for regularization without underfitting

Common Issues
=============

**Training Problems**

1. **Gradient explosion**: Use gradient clipping (max_norm=1.0)
2. **Attention collapse**: Reduce learning rate, add dropout
3. **Poor convergence**: Check data normalization and feature scaling
4. **Overfitting**: Increase dropout, reduce model size, add regularization

**Inference Issues**

1. **Numerical instability**: Use consistent scaling between train/test
2. **Memory errors**: Reduce batch size or sequence length
3. **Slow inference**: Consider model distillation or pruning
4. **Poor generalization**: Augment training data, improve feature engineering

**Performance Tuning**

.. code-block:: python

   # Learning rate scheduling
   scheduler = optim.lr_scheduler.OneCycleLR(
       optimizer, 
       max_lr=0.001,
       epochs=100,
       steps_per_epoch=len(train_loader)
   )
   
   # Mixed precision training
   from torch.cuda.amp import autocast, GradScaler
   
   scaler = GradScaler()
   
   for past_cmas, past_cpps, future_cpps, targets in train_loader:
       optimizer.zero_grad()
       
       with autocast():
           predictions = model(past_cmas, past_cpps, future_cpps)
           loss = criterion(predictions, targets)
       
       scaler.scale(loss).backward()
       scaler.step(optimizer)
       scaler.update()

See Also
========

* :doc:`dataset` - Data preparation for transformer training
* :doc:`mpc_controller` - Using trained models for control
* :doc:`../tutorials/transformer_training` - Step-by-step training guide
* :doc:`../examples/model_analysis` - Advanced model analysis techniques