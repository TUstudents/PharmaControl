import torch
import torch.nn as nn
import math

# We can reuse the PositionalEncoding class from V1
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class ProbabilisticTransformer(nn.Module):
    """
    A Transformer model that supports probabilistic forecasting via MC Dropout.
    The architecture is identical to the V1 predictor, but with added methods.
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
