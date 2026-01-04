# models.py
import torch
import torch.nn as nn

class WaveDeltaCNN(nn.Module):
    """
    Predicts delta (%) to correct the classical estimate:
        y_hat = classic_pct + delta
    """
    def __init__(self, n_channels: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 16, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(2),  # T -> T/2

            nn.Conv1d(16, 32, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(2),  # T/2 -> T/4

            nn.Conv1d(32, 64, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # -> (B,64,1)
        )
        self.head = nn.Sequential(
            nn.Linear(64 + 1, 32),  # + classic_pct conditioning
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x, classic_pct):
        # x: (B,T,C) -> (B,C,T)
        x = x.transpose(1, 2)
        z = self.net(x).squeeze(-1)                      # (B,64)
        z = torch.cat([z, classic_pct.unsqueeze(1)], 1)  # (B,65)
        return self.head(z).squeeze(-1)                  # (B,)
