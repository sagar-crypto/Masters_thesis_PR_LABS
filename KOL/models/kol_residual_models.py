from __future__ import annotations

import torch
import torch.nn as nn

from KOL.common.constants import GROUND_CASE_IDS


class KOLGRUCaseResidualRegressor(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_op_features: int = 0,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
        n_cases: int = 10,
    ):
        super().__init__()
        self.num_dirs = 2 if bidirectional else 1
        self.n_op_features = int(n_op_features)

        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        head_in_dim = hidden_size * self.num_dirs + self.n_op_features + 1

        self.residual_head = nn.Sequential(
            nn.Linear(head_in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_cases),
            nn.Tanh(),
        )

    def forward(
        self,
        x_seq: torch.Tensor,
        case_idx: torch.Tensor,
        d_prior: torch.Tensor,
        op_feat: torch.Tensor,
    ) -> torch.Tensor:
        out, _ = self.gru(x_seq)
        last = out[:, -1, :]

        if op_feat.ndim == 1:
            op_feat = op_feat.unsqueeze(1)

        prior_feat = d_prior.unsqueeze(1)
        fused = torch.cat([last, prior_feat, op_feat], dim=1)

        residual_all = self.residual_head(fused)
        residual = residual_all.gather(1, case_idx.unsqueeze(1)).squeeze(1)
        return residual


def apply_kol_prediction_rule(
    d_phys_prior: torch.Tensor,
    case_idx: torch.Tensor,
    residual: torch.Tensor,
    mode: str = "ground_only_mul",
) -> torch.Tensor:
    mode = str(mode).lower().strip()

    if mode == "prior_only":
        pred = d_phys_prior.clone()

    elif mode == "ground_only_mul":
        pred = d_phys_prior.clone()
        ground_case_ids = torch.tensor(
            sorted(GROUND_CASE_IDS),
            device=case_idx.device,
            dtype=case_idx.dtype,
        )
        mask_ground = (case_idx.unsqueeze(1) == ground_case_ids.unsqueeze(0)).any(dim=1)
        pred[mask_ground] = pred[mask_ground] * (1.0 + residual[mask_ground])

    elif mode == "all_cases_mul":
        pred = d_phys_prior * (1.0 + residual)

    elif mode == "all_cases_add":
        pred = d_phys_prior + residual

    else:
        raise ValueError(
            f"Unknown KOL prediction mode '{mode}'. "
            f"Supported: prior_only, ground_only_mul, all_cases_mul, all_cases_add"
        )

    pred = torch.clamp(pred, 0.0, 1.0)
    return pred
