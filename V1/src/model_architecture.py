import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer-based sequence models.

    Implements the standard sinusoidal positional encoding from "Attention Is All You Need"
    to provide position-dependent information to the transformer architecture. This enables
    the model to understand temporal relationships in process time series data without
    relying on recurrent connections.

    The encoding uses sine and cosine functions of different frequencies to create
    unique positional representations that allow the model to extrapolate to
    sequence lengths not seen during training.

    Args:
        d_model: Model dimension (embedding size) that must match transformer architecture
        dropout: Dropout probability applied after adding positional encodings (default: 0.1)
        max_len: Maximum sequence length supported (default: 5000)

    Attributes:
        dropout: Dropout layer for regularization
        pe: Registered buffer containing precomputed positional encodings

    Notes:
        - Uses sine for even dimensions, cosine for odd dimensions
        - Encodings are cached as registered buffers for efficiency
        - Compatible with variable-length sequences up to max_len
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # FIXED: Create positional encoding for batch_first=True format
        pe = torch.zeros(1, max_len, d_model)  # Shape: (1, max_len, d_model) for batch_first=True
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """Apply positional encoding to input embeddings.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model) containing
               embedded sequence data (e.g., process variables or control actions)
               Note: batch_first=True format as used by the transformer

        Returns:
            Tensor of same shape as input with positional information added
            and dropout applied for regularization during training
        """
        # FIXED: Correct indexing for batch_first=True format
        # x.size(1) is seq_len when batch_first=True
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class GranulationPredictor(nn.Module):
    """Transformer-based neural network for pharmaceutical granulation process prediction.

    This encoder-decoder transformer architecture is specifically designed for predicting
    Critical Material Attributes (CMAs) in continuous granulation processes based on
    historical process data and planned future control actions.

    Architecture Overview:
    - Encoder: Processes historical CMA and CPP time series to build process context
    - Decoder: Generates future CMA predictions conditioned on planned CPP sequences
    - Multi-head attention: Captures complex temporal dependencies and interactions
    - Positional encoding: Maintains temporal order information

    The model implements a sequence-to-sequence prediction paradigm where:
    - Input: Historical CMAs + CPPs (lookback window) + Future CPPs (control plan)
    - Output: Future CMAs (prediction horizon)

    This architecture enables Model Predictive Control by providing multi-step-ahead
    predictions necessary for optimization-based control strategies.

    Args:
        cma_features: Number of Critical Material Attributes (output variables)
            Typically 2 for granulation: particle size (d50) and moisture (LOD)
        cpp_features: Number of Critical Process Parameters including soft sensors
            Typically 5: spray_rate, air_flow, carousel_speed, specific_energy, froude_number
        d_model: Transformer model dimension for embeddings and hidden states (default: 64)
        nhead: Number of attention heads in multi-head attention (default: 4)
        num_encoder_layers: Number of transformer encoder layers (default: 2)
        num_decoder_layers: Number of transformer decoder layers (default: 2)
        dim_feedforward: Hidden dimension in feedforward networks (default: 256)
        dropout: Dropout probability for regularization (default: 0.1)

    Attributes:
        cma_encoder_embedding: Linear layer projecting CMA features to d_model
        cpp_encoder_embedding: Linear layer projecting CPP features to d_model (encoder)
        cpp_decoder_embedding: Linear layer projecting CPP features to d_model (decoder)
        pos_encoder: Positional encoding module for temporal information
        transformer: Core transformer architecture with encoder-decoder structure
        output_linear: Final projection layer mapping d_model back to CMA features

    Example:
        >>> model = GranulationPredictor(cma_features=2, cpp_features=5)
        >>> past_cmas = torch.randn(32, 36, 2)  # (batch, lookback, features)
        >>> past_cpps = torch.randn(32, 36, 5)
        >>> future_cpps = torch.randn(32, 10, 5)  # (batch, horizon, features)
        >>> predictions = model(past_cmas, past_cpps, future_cpps)
        >>> print(predictions.shape)  # torch.Size([32, 10, 2])
    """

    def __init__(
        self,
        cma_features,
        cpp_features,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # --- Input Embeddings ---
        self.cma_encoder_embedding = nn.Linear(cma_features, d_model)
        self.cpp_encoder_embedding = nn.Linear(cpp_features, d_model)
        self.cpp_decoder_embedding = nn.Linear(cpp_features, d_model)

        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # --- Transformer ---
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # --- Output Layer ---
        # Maps the decoder output back to the desired number of CMA features
        self.output_linear = nn.Linear(d_model, cma_features)

    def forward(self, past_cmas, past_cpps, future_cpps):
        """Execute forward pass for granulation process prediction.

        Implements the sequence-to-sequence transformer architecture for multi-step-ahead
        prediction of critical material attributes based on historical process data
        and planned future control actions.

        Forward Pass Architecture:
        1. Embed historical CMAs and CPPs separately, then combine with addition
        2. Apply positional encoding to maintain temporal sequence information
        3. Process through transformer encoder to build process context representation
        4. Embed future CPPs and apply positional encoding for decoder input
        5. Generate causal attention mask to prevent information leakage
        6. Decode predictions using transformer decoder with attention to encoder context
        7. Project decoder output to CMA space using final linear layer

        Args:
            past_cmas: Historical critical material attributes tensor of shape
                (batch_size, lookback_length, cma_features). Contains time series
                of process outputs (e.g., particle size, moisture content)
            past_cpps: Historical critical process parameters tensor of shape
                (batch_size, lookback_length, cpp_features). Contains time series
                of control inputs and soft sensors
            future_cpps: Planned future control actions tensor of shape
                (batch_size, prediction_horizon, cpp_features). Control sequence
                for which predictions are desired

        Returns:
            Tensor of shape (batch_size, prediction_horizon, cma_features) containing
            predicted critical material attributes over the specified horizon.
            Values are in the same scale/units as the training data.

        Notes:
            - Uses causal masking in decoder to prevent future information leakage
            - Additive combination of CMA and CPP embeddings assumes feature alignment
            - Positional encoding enables attention mechanism to utilize temporal order
            - All tensors must be on the same device for computation
        """
        # src: source sequence to the encoder (historical data)
        # tgt: target sequence to the decoder (planned future actions)

        # Embed and combine historical inputs for the encoder
        past_cma_emb = self.cma_encoder_embedding(past_cmas)
        past_cpp_emb = self.cpp_encoder_embedding(past_cpps)
        src = self.pos_encoder(past_cma_emb + past_cpp_emb)

        # Embed future control actions for the decoder
        tgt = self.pos_encoder(self.cpp_decoder_embedding(future_cpps))

        # The decoder needs a target mask to prevent it from seeing future positions
        # when making a prediction at the current position.
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

        # Pass through the transformer
        output = self.transformer(src, tgt, tgt_mask=tgt_mask)

        # Final linear layer to get CMA predictions
        prediction = self.output_linear(output)

        return prediction
