# models.py
import torch
import torch.nn as nn

class WaveDeltaCNN(nn.Module):
    """
    Predicts delta (%) to correct the classical estimate:
        y_hat = classic_pct + delta
    """
    def __init__(self, n_channels: int, context_dim: int, dropout: float = 0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # -> (B,128,1)
        )

        self.head = nn.Sequential(
            nn.Linear(128 + context_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x_win: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x_win: (B,T,C) -> Conv1d expects (B,C,T)
        x = x_win.transpose(1, 2)              # (B,C,T)
        z = self.cnn(x).squeeze(-1)            # (B,128)
        h = torch.cat([z, context], dim=1)     # (B,128+context_dim)
        delta = self.head(h).squeeze(-1)       # (B,)
        return delta                # (B,)
