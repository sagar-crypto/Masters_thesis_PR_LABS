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



class KOLGRUBoundedResidualFusionRegressor(nn.Module):
    """Predict a gated, case-aware bounded correction to a physics prior.

    Inputs use normalized fault distance: waveform ``(batch, time, features)``,
    prior ``(batch,)``, integer fault-case IDs, and optional operator features.
    A GRU representation, prior, case embedding, and normalized operator vector
    feed shared residual/gate heads. ``tanh`` bounds the proposal to
    ``[-residual_max, residual_max]`` and a sigmoid gate attenuates it. The
    configured KOL rule decides whether that residual is additive or
    multiplicative for each fault family. Forward returns clipped prediction,
    unclipped prediction for loss, effective residual, and gate diagnostics.
    """

    def __init__(
        self,
        *,
        n_features: int,
        n_op_features: int = 0,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        n_cases: int = 10,
        case_emb_dim: int = 8,
        head_hidden_size: int = 64,
        residual_max: float = 1.0,
        gate_init_bias: float = -3.0,
        prediction_mode: str = "threeph_add_ground_mul",
    ) -> None:
        super().__init__()

        self.bidirectional = bool(bidirectional)
        self.num_dirs = 2 if self.bidirectional else 1
        self.n_op_features = int(n_op_features)

        self.residual_max = float(residual_max)
        self.prediction_mode = str(
            prediction_mode
        ).lower().strip()

        if self.residual_max <= 0.0:
            raise ValueError(
                "residual_max must be positive, "
                f"got {self.residual_max}"
            )

        self.gru = nn.GRU(
            input_size=int(n_features),
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            dropout=(
                float(dropout)
                if int(num_layers) > 1
                else 0.0
            ),
            batch_first=True,
            bidirectional=self.bidirectional,
        )

        h_dim = (
            int(hidden_size)
            * self.num_dirs
        )

        self.case_emb = nn.Embedding(
            num_embeddings=int(n_cases),
            embedding_dim=int(case_emb_dim),
        )

        if self.n_op_features > 0:
            self.op_norm = nn.BatchNorm1d(
                self.n_op_features
            )
        else:
            self.op_norm = None

        # Inputs:
        # GRU hidden state
        # physics prior
        # fault-case embedding
        fusion_in_dim = (
            h_dim
            + 1
            + int(case_emb_dim)
            + self.n_op_features
        )

        head_hidden_size = int(
            head_hidden_size
        )

        self.shared_head = nn.Sequential(
            nn.Linear(
                fusion_in_dim,
                head_hidden_size,
            ),
            nn.ReLU(),
            nn.Linear(
                head_hidden_size,
                max(
                    head_hidden_size // 2,
                    8,
                ),
            ),
            nn.ReLU(),
        )

        head_out_dim = max(
            head_hidden_size // 2,
            8,
        )

        self.residual_head = nn.Linear(
            head_out_dim,
            1,
        )

        self.gate_head = nn.Linear(
            head_out_dim,
            1,
        )

        # Start exactly from the physics prior.
        nn.init.zeros_(
            self.residual_head.weight
        )
        nn.init.zeros_(
            self.residual_head.bias
        )

        nn.init.zeros_(
            self.gate_head.weight
        )
        nn.init.constant_(
            self.gate_head.bias,
            float(gate_init_bias),
        )

    def forward(
        self,
        x_seq: torch.Tensor,
        d_phys_prior: torch.Tensor,
        case_idx: torch.Tensor,
        op_features: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        out, _ = self.gru(
            x_seq
        )

        h = out[:, -1, :]

        case_embedding = self.case_emb(
            case_idx.long()
        )

        fusion_pieces = [
            h,
            d_phys_prior.unsqueeze(1),
            case_embedding,
        ]

        if self.n_op_features > 0:
            if op_features is None:
                raise ValueError(
                    "op_features is required because "
                    "n_op_features > 0"
                )

            if op_features.ndim == 1:
                op_features = op_features.unsqueeze(1)

            if op_features.shape[1] != self.n_op_features:
                raise ValueError(
                    "Unexpected operator-feature width: "
                    f"received {op_features.shape[1]}, "
                    f"expected {self.n_op_features}"
                )

            op_norm = self.op_norm

            if op_norm is None:
                raise RuntimeError(
                    "op_norm is not initialized although "
                    "n_op_features > 0"
                )

            normalized_op_features = op_norm(
                op_features.float()
            )

            fusion_pieces.append(
                normalized_op_features
            )

        fusion_input = torch.cat(
            fusion_pieces,
            dim=1,
        )

        shared = self.shared_head(
            fusion_input
        )

        residual_proposal = (
            self.residual_max
            * torch.tanh(
                self.residual_head(
                    shared
                )
            ).squeeze(-1)
        )

        gate = torch.sigmoid(
            self.gate_head(
                shared
            )
        ).squeeze(-1)

        # This is the residual passed to the established KOL rule.
        residual = (
            gate
            * residual_proposal
        )

        d_pred_unclipped = (
            apply_kol_prediction_rule_unclipped(
                d_phys_prior=d_phys_prior,
                case_idx=case_idx,
                residual=residual,
                mode=self.prediction_mode,
            )
        )

        d_pred = torch.clamp(
            d_pred_unclipped,
            0.0,
            1.0,
        )

        return (
            d_pred,
            d_pred_unclipped,
            residual,
            gate,
        )



def apply_kol_prediction_rule_unclipped(
    d_phys_prior: torch.Tensor,
    case_idx: torch.Tensor,
    residual: torch.Tensor,
    mode: str = "ground_only_mul",
) -> torch.Tensor:
    """Apply the configured case-aware correction without output clipping.

    ``d_phys_prior`` and additive residual terms use normalized line distance
    (0--1), while multiplicative branches treat the residual as a relative
    correction. ``case_idx`` selects ground, three-phase, and line-line branches;
    modes intentionally leave unlisted cases at the prior. Keeping the raw result
    allows training loss to penalize predictions outside the physical interval.

    Returns:
        A tensor shaped like ``d_phys_prior``. Unknown modes raise ``ValueError``.
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
    """Apply the KOL correction and clip final normalized distances to [0, 1].

    Training can use :func:`apply_kol_prediction_rule_unclipped` for an
    out-of-range penalty; evaluation and exported predictions use this bounded
    form. Inputs and case-dependent additive/multiplicative semantics are
    otherwise identical.
    """
    pred = apply_kol_prediction_rule_unclipped(
        d_phys_prior=d_phys_prior,
        case_idx=case_idx,
        residual=residual,
        mode=mode,
    )

    return torch.clamp(pred, 0.0, 1.0)
