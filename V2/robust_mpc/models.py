import torch
import torch.nn as nn
import math

# CRITICAL FIX: Use corrected PositionalEncoding for batch_first=True
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer-based sequence models.
    
    Implements the standard sinusoidal positional encoding with proper support
    for batch_first=True format used in the V2 transformer architecture.
    
    Args:
        d_model (int): Model dimension (embedding size)
        dropout (float, optional): Dropout probability. Default: 0.1
        max_len (int, optional): Maximum sequence length. Default: 5000
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # FIXED: Create positional encoding for batch_first=True format
        pe = torch.zeros(1, max_len, d_model)  # Shape: (1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Apply positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            Tensor with positional information added and dropout applied
        """
        # FIXED: Correct indexing for batch_first=True format
        # x.size(1) is seq_len when batch_first=True
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class ProbabilisticTransformer(nn.Module):
    """Transformer-based probabilistic neural network for pharmaceutical process prediction.
    
    This model extends the standard transformer architecture with uncertainty quantification
    capabilities through Monte Carlo Dropout, enabling robust Model Predictive Control
    with risk-aware decision making. The architecture maintains the encoder-decoder
    structure optimized for pharmaceutical granulation processes while adding
    distributional prediction capabilities.
    
    Key Features:
        - Probabilistic predictions through Monte Carlo Dropout sampling
        - Uncertainty quantification for risk-aware control decisions
        - Sequence-to-sequence architecture for multi-step ahead forecasting
        - Optimized for Critical Material Attribute prediction in granulation
    
    Architecture:
        - Encoder: Processes historical CMA and CPP time series
        - Decoder: Generates future CMA predictions from planned CPP sequences
        - Attention: Captures complex temporal dependencies and interactions
        - Probabilistic output: Mean and standard deviation estimates
    
    Mathematical Framework:
        Input: [Historical CMAs, Historical CPPs] + [Future CPPs]
        Output: p(Future CMAs | Historical data, Future CPPs)
        
        Uncertainty estimation via Monte Carlo Dropout:
        μ(x) = E[f(x, θ)] ≈ (1/N) Σ f(x, θᵢ)  where θᵢ ~ Dropout
        σ(x) = Var[f(x, θ)] ≈ (1/N) Σ [f(x, θᵢ) - μ(x)]²
    
    Args:
        cma_features (int): Number of Critical Material Attributes (typically 2: d50, LOD)
        cpp_features (int): Number of Critical Process Parameters including soft sensors
            (typically 5: spray_rate, air_flow, carousel_speed, specific_energy, froude_number)
        d_model (int, optional): Transformer model dimension. Default: 64
        nhead (int, optional): Number of attention heads. Default: 4
        num_encoder_layers (int, optional): Number of encoder layers. Default: 2
        num_decoder_layers (int, optional): Number of decoder layers. Default: 2
        dim_feedforward (int, optional): Feedforward network dimension. Default: 256
        dropout (float, optional): Dropout probability for uncertainty quantification. Default: 0.1
    
    Attributes:
        d_model (int): Model dimension for embeddings
        cma_features (int): Number of output variables
        cpp_features (int): Number of input control variables
        cma_encoder_embedding (nn.Linear): CMA feature embedding layer
        cpp_encoder_embedding (nn.Linear): CPP encoder embedding layer
        cpp_decoder_embedding (nn.Linear): CPP decoder embedding layer
        pos_encoder (PositionalEncoding): Positional encoding module
        transformer (nn.Transformer): Core transformer architecture
        output_linear (nn.Linear): Final projection to CMA predictions
    
    Example:
        >>> # Configure for granulation process
        >>> model = ProbabilisticTransformer(
        ...     cma_features=2,      # d50, LOD
        ...     cpp_features=5,      # spray, air, speed + soft sensors
        ...     dropout=0.15         # Higher dropout for better uncertainty
        ... )
        >>> 
        >>> # Probabilistic prediction
        >>> past_cmas = torch.randn(1, 10, 2)    # Historical CMAs
        >>> past_cpps = torch.randn(1, 10, 5)    # Historical CPPs
        >>> future_cpps = torch.randn(1, 5, 5)   # Planned control actions
        >>> mean_pred, std_pred = model.predict_distribution(
        ...     past_cmas, past_cpps, future_cpps, n_samples=50
        ... )
    
    Notes:
        - Uses batch_first=True format for all tensor operations
        - Dropout should be enabled during inference for uncertainty quantification
        - Higher dropout rates improve uncertainty estimates but may reduce accuracy
        - Monte Carlo sampling provides calibrated uncertainty bounds
    """
    def __init__(self, cma_features, cpp_features, d_model=64, nhead=4, 
                 num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.cma_features = cma_features
        self.cpp_features = cpp_features

        # --- Layers (identical to V1 model) ---
        self.cma_encoder_embedding = nn.Linear(cma_features, d_model)
        self.cpp_encoder_embedding = nn.Linear(cpp_features, d_model)
        self.cpp_decoder_embedding = nn.Linear(cpp_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.output_linear = nn.Linear(d_model, cma_features)

    def forward(self, past_cmas, past_cpps, future_cpps):
        past_cma_emb = self.cma_encoder_embedding(past_cmas)
        past_cpp_emb = self.cpp_encoder_embedding(past_cpps)
        src = self.pos_encoder(past_cma_emb + past_cpp_emb)
        tgt = self.pos_encoder(self.cpp_decoder_embedding(future_cpps))
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        output = self.transformer(src, tgt, tgt_mask=tgt_mask)
        return self.output_linear(output)

    def predict_distribution(self, past_cmas, past_cpps, future_cpps, n_samples=30):
        """
        Performs probabilistic forecasting using Monte Carlo Dropout.

        Returns:
            tuple: (mean_prediction, std_prediction)
        """
        # Set to evaluation mode BUT keep dropout layers active.
        # This is done by activating training mode only for dropout layers.
        self.train()
        # An alternative, cleaner way is to create a custom method to only turn on dropout.
        # For simplicity, we use train() as it activates dropout.

        with torch.no_grad():
            # Collect multiple predictions
            predictions = [self.forward(past_cmas, past_cpps, future_cpps) for _ in range(n_samples)]

            # Stack predictions into a new dimension for calculation
            # Shape: (n_samples, batch_size, horizon, features)
            predictions_stacked = torch.stack(predictions)

            # Calculate mean and standard deviation across the samples dimension
            mean_prediction = torch.mean(predictions_stacked, dim=0)
            std_prediction = torch.std(predictions_stacked, dim=0)

        # Return model to standard evaluation mode
        self.eval()

        return mean_prediction, std_prediction
