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


class KOLDualGRUCaseResidualRegressor(nn.Module):
    """
    Dual-branch KOL residual model.

    Branch 1:
        waveform sequence -> GRU

    Branch 2:
        phasor sequence -> GRU

    Then:
        waveform embedding + phasor embedding + case embedding + operator features
        -> MLP
        -> residual correction
    """

    def __init__(
        self,
        *,
        n_waveform_features: int,
        n_phasor_features: int,
        n_op_features: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
        n_cases: int = 10,
        case_emb_dim: int = 8,
    ) -> None:
        super().__init__()

        self.n_op_features = int(n_op_features)
        self.bidirectional = bool(bidirectional)

        self.waveform_gru = nn.GRU(
            input_size=int(n_waveform_features),
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
            bidirectional=bool(bidirectional),
        )

        self.phasor_gru = nn.GRU(
            input_size=int(n_phasor_features),
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
            bidirectional=bool(bidirectional),
        )

        gru_out_dim = int(hidden_size) * (2 if bidirectional else 1)

        self.case_emb = nn.Embedding(
            num_embeddings=int(n_cases),
            embedding_dim=int(case_emb_dim),
        )

        mlp_in_dim = (
            gru_out_dim
            + gru_out_dim
            + int(case_emb_dim)
            + int(n_op_features)
        )

        self.head = nn.Sequential(
            nn.Linear(mlp_in_dim, int(hidden_size)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_size), int(hidden_size) // 2),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_size) // 2, 1),
        )

    def _last_hidden(self, h_n: torch.Tensor) -> torch.Tensor:
        """
        h_n shape:
            unidirectional: (num_layers, B, H)
            bidirectional: (num_layers * 2, B, H)
        """
        if not self.bidirectional:
            return h_n[-1]

        # Last layer forward and backward states
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        return torch.cat([h_forward, h_backward], dim=-1)

    def forward(
        self,
        x_waveform: torch.Tensor,
        x_phasor: torch.Tensor,
        d_prior: torch.Tensor,
        case_idx: torch.Tensor,
        op_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, h_waveform = self.waveform_gru(x_waveform)
        _, h_phasor = self.phasor_gru(x_phasor)

        h_waveform_last = self._last_hidden(h_waveform)
        h_phasor_last = self._last_hidden(h_phasor)

        case_idx = case_idx.long()
        case_embedding = self.case_emb(case_idx)

        pieces = [h_waveform_last, h_phasor_last, case_embedding]

        if self.n_op_features > 0:
            if op_features is None:
                raise ValueError("op_features is required but got None.")
            pieces.append(op_features.float())

        combined = torch.cat(pieces, dim=-1)

        residual = self.head(combined).squeeze(-1)

        return residual



def apply_kol_prediction_rule_unclipped(
    d_phys_prior: torch.Tensor,
    case_idx: torch.Tensor,
    residual: torch.Tensor,
    mode: str = "ground_only_mul",
) -> torch.Tensor:
    """
    Same correction rule as apply_kol_prediction_rule, but without final clipping.

    This is useful for training loss because the model should be penalized
    if the raw prediction goes below 0 or above 1.
    """
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

        pred[mask_ground] = d_phys_prior[mask_ground] * (
            1.0 + residual[mask_ground]
        )

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

        pred[mask_ground] = d_phys_prior[mask_ground] * (
            1.0 + residual[mask_ground]
        )
        pred[mask_non_ground] = d_phys_prior[mask_non_ground] + (
            0.25 * residual[mask_non_ground]
        )

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

        pred[mask_ground] = d_phys_prior[mask_ground] * (
            1.0 + residual[mask_ground]
        )
        pred[mask_3ph] = d_phys_prior[mask_3ph] + (
            0.50 * residual[mask_3ph]
        )

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

        pred[mask_ground] = d_phys_prior[mask_ground] * (
            1.0 + residual[mask_ground]
        )
        pred[mask_3ph] = d_phys_prior[mask_3ph] + (
            0.50 * residual[mask_3ph]
        )
        pred[mask_ll] = d_phys_prior[mask_ll] + (
            0.10 * residual[mask_ll]
        )

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

        pred[mask_ground] = d_phys_prior[mask_ground] * (
            1.0 + residual[mask_ground]
        )
        pred[mask_3ph] = d_phys_prior[mask_3ph] + (
            0.50 * residual[mask_3ph]
        )
        pred[mask_ll] = d_phys_prior[mask_ll] + (
            0.05 * residual[mask_ll]
        )

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

        pred[mask_ground] = d_phys_prior[mask_ground] * (
            1.0 + residual[mask_ground]
        )
        pred[mask_3ph] = d_phys_prior[mask_3ph] + (
            0.50 * residual[mask_3ph]
        )
        pred[mask_llca] = d_phys_prior[mask_llca] + (
            0.05 * residual[mask_llca]
        )

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

        pred[mask_ground] = d_phys_prior[mask_ground] * (
            1.0 + 0.25 * residual[mask_ground]
        )
        pred[mask_3ph] = d_phys_prior[mask_3ph] + (
            0.25 * residual[mask_3ph]
        )

    elif mode == "threeph_add_ground_mul_tanh_050":
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

        bounded_residual = torch.tanh(residual)

        pred[mask_ground] = d_phys_prior[mask_ground] * (
            1.0 + 0.50 * bounded_residual[mask_ground]
        )

        pred[mask_3ph] = d_phys_prior[mask_3ph] + (
            0.50 * bounded_residual[mask_3ph]
        )

    elif mode == "threeph_add_ground_mul_tanh_100":
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

        bounded_residual = torch.tanh(residual)

        pred[mask_ground] = d_phys_prior[mask_ground] * (
            1.0 + bounded_residual[mask_ground]
        )

        pred[mask_3ph] = d_phys_prior[mask_3ph] + (
            1.0 * bounded_residual[mask_3ph]
        )

    elif mode == "all_cases_add_025":
        pred = d_phys_prior + 0.25 * residual

    elif mode == "all_cases_add_050":
        pred = d_phys_prior + 0.50 * residual

    elif mode == "bounded_add_030":
        pred = d_phys_prior + 0.30 * torch.tanh(residual)

    elif mode == "bounded_add_050":
        pred = d_phys_prior + 0.50 * torch.tanh(residual)

    else:
        raise ValueError(
            f"Unknown KOL prediction mode '{mode}'. "
            f"Supported: prior_only, ground_only_mul, all_cases_mul, all_cases_add, "
            f"threeph_add_ground_mul, familywise_add_mul, all_cases_add_010, "
            f"threeph_add_ground_mul_025, all_cases_add_025, all_cases_add_050, "
            f"bounded_add_030, bounded_add_050"
        )

    return pred


def apply_kol_prediction_rule(
    d_phys_prior: torch.Tensor,
    case_idx: torch.Tensor,
    residual: torch.Tensor,
    mode: str = "ground_only_mul",
) -> torch.Tensor:
    """
    Final prediction rule used for evaluation/output.

    This version clips predictions to the valid normalized fault-location range [0, 1].
    """
    pred = apply_kol_prediction_rule_unclipped(
        d_phys_prior=d_phys_prior,
        case_idx=case_idx,
        residual=residual,
        mode=mode,
    )

    return torch.clamp(pred, 0.0, 1.0)
