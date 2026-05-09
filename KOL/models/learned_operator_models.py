from __future__ import annotations

import torch
import torch.nn as nn


class LearnedOperatorGRU(nn.Module):
    def __init__(
        self,
        n_features: int = 6,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
        n_cases: int = 10,
    ):
        super().__init__()
        self.num_dirs = 2 if bidirectional else 1
        self.n_cases = n_cases

        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size * self.num_dirs, 128),
            nn.ReLU(),
            nn.Linear(128, 4 * n_cases),
            nn.Tanh(),
        )

    def forward(
        self,
        x_seq: torch.Tensor,
        case_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out, _ = self.gru(x_seq)
        last = out[:, -1, :]
        pred_all = self.head(last).view(x_seq.shape[0], self.n_cases, 4)

        selected = pred_all.gather(
            1,
            case_idx.view(-1, 1, 1).expand(-1, 1, 4)
        ).squeeze(1)

        delta_alpha_re = 0.2 * selected[:, 0]
        delta_alpha_im = 0.2 * selected[:, 1]
        delta_k0_re = 0.5 * selected[:, 2]
        delta_k0_im = 0.5 * selected[:, 3]

        return delta_alpha_re, delta_alpha_im, delta_k0_re, delta_k0_im
    


