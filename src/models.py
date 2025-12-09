"""Model definitions for forecasting (LSTM baseline)."""

import torch
import torch.nn as nn

__all__ = ["LSTMForecaster"]


class LSTMForecaster(nn.Module):
    """
    Simple multivariate LSTM forecaster with a dense head for multi-step output.

    Input:  (batch, seq_len, n_features)
    Output: (batch, horizon)
    """

    def __init__(self, input_dim: int, hidden_size: int = 64, num_layers: int = 1, horizon: int = 24, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use the last hidden state to predict the full horizon in one shot
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)
