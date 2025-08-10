import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Injects positional information into the input sequence."""
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

class GranulationPredictor(nn.Module):
    """
    A Transformer-based Encoder-Decoder model for predicting granulation CMAs.
    """
    def __init__(self, cma_features, cpp_features, d_model=64, nhead=4, 
                 num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=256, dropout=0.1):
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
            batch_first=True
        )

        # --- Output Layer ---
        # Maps the decoder output back to the desired number of CMA features
        self.output_linear = nn.Linear(d_model, cma_features)

    def forward(self, past_cmas, past_cpps, future_cpps):
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
