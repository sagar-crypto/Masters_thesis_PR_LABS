from __future__ import annotations

import torch
import torch.nn as nn

from KOL.common.constants import GROUND_CASE_IDS, CASE_TO_IDX


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

    elif mode == "non_ground_add_ground_mul":
        pred = d_phys_prior.clone()

        ground_case_ids = torch.tensor(
            sorted(GROUND_CASE_IDS),
            device=case_idx.device,
            dtype=case_idx.dtype,
        )
        mask_ground = (case_idx.unsqueeze(1) == ground_case_ids.unsqueeze(0)).any(dim=1)
        mask_non_ground = ~mask_ground

        # Keep current stable behavior for ground faults
        pred[mask_ground] = d_phys_prior[mask_ground] * (1.0 + residual[mask_ground])

        # Allow controlled additive correction for 3ph and LL faults
        pred[mask_non_ground] = d_phys_prior[mask_non_ground] + 0.25 * residual[mask_non_ground]

    elif mode == "threeph_add_ground_mul":
        pred = d_phys_prior.clone()

        ground_case_ids = torch.tensor(
            sorted(GROUND_CASE_IDS),
            device=case_idx.device,
            dtype=case_idx.dtype,
        )
        mask_ground = (case_idx.unsqueeze(1) == ground_case_ids.unsqueeze(0)).any(dim=1)

        case_3ph_id = torch.tensor(
            CASE_TO_IDX["3ph"],
            device=case_idx.device,
            dtype=case_idx.dtype,
        )
        mask_3ph = case_idx == case_3ph_id

        # Keep the successful behavior for ground faults
        pred[mask_ground] = d_phys_prior[mask_ground] * (1.0 + residual[mask_ground])

        # Allow only 3ph to move additively
        pred[mask_3ph] = d_phys_prior[mask_3ph] + 0.50 * residual[mask_3ph]


    elif mode == "familywise_add_mul":
        pred = d_phys_prior.clone()

        ground_case_ids = torch.tensor(
            sorted(GROUND_CASE_IDS),
            device=case_idx.device,
            dtype=case_idx.dtype,
        )
        mask_ground = (case_idx.unsqueeze(1) == ground_case_ids.unsqueeze(0)).any(dim=1)

        case_3ph_id = torch.tensor(
            CASE_TO_IDX["3ph"],
            device=case_idx.device,
            dtype=case_idx.dtype,
        )
        mask_3ph = case_idx == case_3ph_id

        ll_case_ids = torch.tensor(
            [
                CASE_TO_IDX["ll_ab"],
                CASE_TO_IDX["ll_bc"],
                CASE_TO_IDX["ll_ca"],
            ],
            device=case_idx.device,
            dtype=case_idx.dtype,
        )
        mask_ll = (case_idx.unsqueeze(1) == ll_case_ids.unsqueeze(0)).any(dim=1)

        # Ground faults: keep the stable multiplicative correction
        pred[mask_ground] = d_phys_prior[mask_ground] * (1.0 + residual[mask_ground])

        # 3ph: current best additive correction scale
        pred[mask_3ph] = d_phys_prior[mask_3ph] + 0.50 * residual[mask_3ph]

        # LL faults: small additive correction
        pred[mask_ll] = d_phys_prior[mask_ll] + 0.10 * residual[mask_ll]

    elif mode == "all_faults_casewise_safe":
        pred = d_phys_prior.clone()

        ground_case_ids = torch.tensor(
            sorted(GROUND_CASE_IDS),
            device=case_idx.device,
            dtype=case_idx.dtype,
        )
        mask_ground = (case_idx.unsqueeze(1) == ground_case_ids.unsqueeze(0)).any(dim=1)

        mask_3ph = case_idx == torch.tensor(
            CASE_TO_IDX["3ph"],
            device=case_idx.device,
            dtype=case_idx.dtype,
        )

        ll_case_ids = torch.tensor(
            [
                CASE_TO_IDX["ll_ab"],
                CASE_TO_IDX["ll_bc"],
                CASE_TO_IDX["ll_ca"],
            ],
            device=case_idx.device,
            dtype=case_idx.dtype,
        )
        mask_ll = (case_idx.unsqueeze(1) == ll_case_ids.unsqueeze(0)).any(dim=1)

        # Keep the current good ground-fault correction
        pred[mask_ground] = d_phys_prior[mask_ground] * (1.0 + residual[mask_ground])

        # Keep the current best 3ph correction
        pred[mask_3ph] = d_phys_prior[mask_3ph] + 0.50 * residual[mask_3ph]

        # Newly added: very conservative LL correction
        pred[mask_ll] = d_phys_prior[mask_ll] + 0.05 * residual[mask_ll]
    
    elif mode == "threeph_ground_llca_add":
        pred = d_phys_prior.clone()

        ground_case_ids = torch.tensor(
            sorted(GROUND_CASE_IDS),
            device=case_idx.device,
            dtype=case_idx.dtype,
        )
        mask_ground = (case_idx.unsqueeze(1) == ground_case_ids.unsqueeze(0)).any(dim=1)

        mask_3ph = case_idx == torch.tensor(
            CASE_TO_IDX["3ph"],
            device=case_idx.device,
            dtype=case_idx.dtype,
        )

        mask_llca = case_idx == torch.tensor(
            CASE_TO_IDX["ll_ca"],
            device=case_idx.device,
            dtype=case_idx.dtype,
        )

        # Keep existing best corrections
        pred[mask_ground] = d_phys_prior[mask_ground] * (1.0 + residual[mask_ground])
        pred[mask_3ph] = d_phys_prior[mask_3ph] + 0.50 * residual[mask_3ph]

        # Only correct the worst LL case, very conservatively
        pred[mask_llca] = d_phys_prior[mask_llca] + 0.05 * residual[mask_llca]

    elif mode == "all_cases_add_010":
        pred = d_phys_prior + 0.10 * residual

    elif mode == "threeph_add_ground_mul_025":
        pred = d_phys_prior.clone()

        ground_case_ids = torch.tensor(
            sorted(GROUND_CASE_IDS),
            device=case_idx.device,
            dtype=case_idx.dtype,
        )
        mask_ground = (case_idx.unsqueeze(1) == ground_case_ids.unsqueeze(0)).any(dim=1)

        case_3ph_id = torch.tensor(
            CASE_TO_IDX["3ph"],
            device=case_idx.device,
            dtype=case_idx.dtype,
        )
        mask_3ph = case_idx == case_3ph_id

        # Smaller correction because Takagi-local prior is already stronger
        pred[mask_ground] = d_phys_prior[mask_ground] * (
            1.0 + 0.25 * residual[mask_ground]
        )

        pred[mask_3ph] = d_phys_prior[mask_3ph] + 0.25 * residual[mask_3ph]

    elif mode == "all_cases_add_025":
        pred = d_phys_prior + 0.25 * residual

    elif mode == "all_cases_add_050":
        pred = d_phys_prior + 0.50 * residual

    else:
        raise ValueError(
            f"Unknown KOL prediction mode '{mode}'. "
            f"Supported: prior_only, ground_only_mul, all_cases_mul, all_cases_add"
        )

    pred = torch.clamp(pred, 0.0, 1.0)
    return pred
