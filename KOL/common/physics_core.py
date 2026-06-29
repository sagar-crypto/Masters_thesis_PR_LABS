from __future__ import annotations

import numpy as np
import torch

from KOL.common.signal_ops import (
    dft_phasor_1cycle,
    symm_pos_seq,
    symm_zero_seq,
    symm_neg_seq,
)
from KOL.common.constants import ALL_CASE_TO_IDX

def k0_from_line(r1: float, x1: float, r0: float, x0: float) -> complex:
    z1 = complex(r1, x1)
    z0 = complex(r0, x0)
    if abs(z1) < 1e-12:
        return 0j
    return (z0 - z1) / (z1)


def compute_two_ended_posseq_distance_pct(
    x_vi_local: np.ndarray,
    x_vi_remote: np.ndarray,
    fs: float,
    f_nom: float,
    r1: float,
    x1: float,
    current_sign: int = 1,
    min_current_ratio: float = 1e-2,
) -> tuple[float, str]:
    """
    Synchronized two-ended positive-sequence fault-location estimate.

    Expected input order per terminal:
        [Va, Vb, Vc, Ia, Ib, Ic]

    Uses the last full cycle of the selected window, matching the
    supervisor's implementation. r1 and x1 are the total impedance
    of the complete protected line in this repository.

    Returns distance in percent from the default/local terminal.
    No clipping is applied.
    """
    if (
        x_vi_local.ndim != 2
        or x_vi_remote.ndim != 2
        or x_vi_local.shape[1] != 6
        or x_vi_remote.shape[1] != 6
    ):
        return np.nan, "expected_two_Tx6_inputs"

    spc = int(np.rint(fs / f_nom))

    if spc <= 1:
        return np.nan, "invalid_samples_per_cycle"

    if x_vi_local.shape[0] < spc or x_vi_remote.shape[0] < spc:
        return np.nan, "window_shorter_than_one_cycle"

    def positive_sequence_vi(x_vi: np.ndarray) -> tuple[complex, complex]:
        cycle = x_vi[-spc:, :]

        va = dft_phasor_1cycle(cycle[:, 0])
        vb = dft_phasor_1cycle(cycle[:, 1])
        vc = dft_phasor_1cycle(cycle[:, 2])

        ia = dft_phasor_1cycle(cycle[:, 3])
        ib = dft_phasor_1cycle(cycle[:, 4])
        ic = dft_phasor_1cycle(cycle[:, 5])

        return (
            symm_pos_seq(va, vb, vc),
            symm_pos_seq(ia, ib, ic),
        )

    v_local, i_local = positive_sequence_vi(x_vi_local)
    v_remote, i_remote = positive_sequence_vi(x_vi_remote)

    if not all(
        np.isfinite(value)
        for value in (
            v_local.real, v_local.imag,
            i_local.real, i_local.imag,
            v_remote.real, v_remote.imag,
            i_remote.real, i_remote.imag,
        )
    ):
        return np.nan, "nonfinite_positive_sequence_phasor"

    z1_total = complex(r1, x1)

    if abs(z1_total) < 1e-12:
        return np.nan, "invalid_z1_total"

    i_local = int(current_sign) * i_local
    i_remote = int(current_sign) * i_remote

    denominator_current = i_local + i_remote
    current_scale = max(abs(i_local), abs(i_remote), 1e-30)

    if abs(denominator_current) < min_current_ratio * current_scale:
        return np.nan, "degenerate_current_denominator"

    distance_pu = (
        v_local - v_remote + z1_total * i_remote
    ) / (
        z1_total * denominator_current
    )

    distance_pct = 100.0 * float(np.real(distance_pu))

    if not np.isfinite(distance_pct):
        return np.nan, "nonfinite_distance"

    return distance_pct, "ok"


def k0_from_line_torch(
    r1: torch.Tensor,
    x1: torch.Tensor,
    r0: torch.Tensor,
    x0: torch.Tensor,
) -> torch.Tensor:
    z1 = torch.complex(r1, x1)
    z0 = torch.complex(r0, x0)
    eps = 1e-12
    z1_safe = torch.where(
        torch.abs(z1) < eps,
        torch.full_like(z1, eps + 0j),
        z1,
    )
    return (z0 - z1) / z1_safe


def compute_classical_distance(
    z_app: complex,
    r1: float,
    x1: float,
    line_len_km: float,
    mode: str = "abs",
) -> float:
    z1 = complex(r1, x1)
    if abs(z1) < 1e-12 or not np.isfinite(line_len_km):
        return np.nan

    if mode == "abs":
        d = (abs(z_app) / abs(z1)) * float(line_len_km)
    elif mode == "real":
        d = np.real(z_app / z1) * float(line_len_km)
    else:
        raise ValueError(f"Unknown distance mode: {mode}")

    return float(np.clip(d, 0.0, float(line_len_km)))



def clip_pct_with_flags(raw_pct: float) -> tuple[float, int, int]:
    clipped = float(np.clip(raw_pct, 0.0, 100.0))
    is_low = int(raw_pct < 0.0)
    is_high = int(raw_pct > 100.0)
    return clipped, is_low, is_high


def edge_distance_score(pct: float) -> float:
    return float(min(abs(pct - 0.0), abs(100.0 - pct)))


def edge_gated_fusion(
    d_local_pct: float,
    d_remote_flipped_pct: float,
    disagreement_threshold_pct: float = 25.0,
) -> float:
    if not (np.isfinite(d_local_pct) and np.isfinite(d_remote_flipped_pct)):
        return np.nan

    diff = abs(d_local_pct - d_remote_flipped_pct)
    if diff <= disagreement_threshold_pct:
        return 0.5 * (d_local_pct + d_remote_flipped_pct)

    local_edge = edge_distance_score(d_local_pct)
    remote_edge = edge_distance_score(d_remote_flipped_pct)
    return float(d_local_pct if local_edge <= remote_edge else d_remote_flipped_pct)


def confidence_weight_from_clip_flags(is_low: int, is_high: int) -> float:
    return 0.25 if (is_low == 1 or is_high == 1) else 1.0


def weighted_fusion_from_confidence(
    d_local_pct: float,
    d_remote_flipped_pct: float,
    w_local: float,
    w_remote: float,
) -> float:
    denom = w_local + w_remote
    if denom <= 1e-12:
        return 0.5 * (d_local_pct + d_remote_flipped_pct)
    return float((w_local * d_local_pct + w_remote * d_remote_flipped_pct) / denom)


def compute_zapp_from_window(
    x_raw: np.ndarray,
    fs: float,
    f_nom: float,
    r1: float,
    x1: float,
    r0: float,
    x0: float,
    case: str,
    dt_start: float,
    onset_idx_from_dt_start_fn,
) -> tuple[complex, str, str]:
    if x_raw.shape[1] not in {6, 12}:
        return np.nan + 1j * np.nan, "invalid", f"unexpected_channel_count_{x_raw.shape[1]}"

    if x_raw.shape[1] == 12:
        x_raw = x_raw[:, :6]

    spc = int(np.rint(fs / f_nom))
    if spc <= 1:
        return np.nan + 1j * np.nan, "invalid", "invalid_spc"

    onset_idx = onset_idx_from_dt_start_fn(dt_start, fs)

    pre_start = onset_idx - spc
    post_start = onset_idx
    post_end = onset_idx + spc

    if pre_start < 0:
        return np.nan + 1j * np.nan, case, "pre_window_out_of_bounds"
    if post_end > x_raw.shape[0]:
        return np.nan + 1j * np.nan, case, "post_window_out_of_bounds"

    Va_po = dft_phasor_1cycle(x_raw[post_start:post_end, 0])
    Vb_po = dft_phasor_1cycle(x_raw[post_start:post_end, 1])
    Vc_po = dft_phasor_1cycle(x_raw[post_start:post_end, 2])
    Ia_po = dft_phasor_1cycle(x_raw[post_start:post_end, 3])
    Ib_po = dft_phasor_1cycle(x_raw[post_start:post_end, 4])
    Ic_po = dft_phasor_1cycle(x_raw[post_start:post_end, 5])

    eps = 1e-9
    i0_po = Ia_po + Ib_po + Ic_po
    k0 = k0_from_line(r1, x1, r0, x0)/3.0

    if case == "3ph":
        v1_po = symm_pos_seq(Va_po, Vb_po, Vc_po)
        i1_po = symm_pos_seq(Ia_po, Ib_po, Ic_po)
        z_po = v1_po / (i1_po + eps)
    elif case == "slg_a":
        z_po = Va_po / (Ia_po + k0 * i0_po + eps)
    elif case == "slg_b":
        z_po = Vb_po / (Ib_po + k0 * i0_po + eps)
    elif case == "slg_c":
        z_po = Vc_po / (Ic_po + k0 * i0_po + eps)
    elif case == "ll_ab":
        z_po = (Va_po - Vb_po) / ((Ia_po - Ib_po) + eps)
    elif case == "ll_bc":
        z_po = (Vb_po - Vc_po) / ((Ib_po - Ic_po) + eps)
    elif case == "ll_ca":
        z_po = (Vc_po - Va_po) / ((Ic_po - Ia_po) + eps)
    elif case == "llg_ab":
        z_po = (Va_po - Vb_po) / ((Ia_po - Ib_po) + eps)
    elif case == "llg_bc":
        z_po = (Vb_po - Vc_po) / ((Ib_po - Ic_po) + eps)
    elif case == "llg_ca":
        z_po = (Vc_po - Va_po) / ((Ic_po - Ia_po) + eps)
    else:
        return np.nan + 1j * np.nan, case, "unknown_case"

    if not (np.isfinite(np.real(z_po)) and np.isfinite(np.imag(z_po))):
        return np.nan + 1j * np.nan, case, "zapp_not_finite"

    return z_po, case, "ok"


def dft_phasor_1cycle_torch(x: torch.Tensor) -> torch.Tensor:
    """
    x: (..., N) real tensor
    returns: (...) complex tensor
    """
    n = x.shape[-1]
    if n <= 0:
        return torch.zeros(x.shape[:-1], dtype=torch.complex64, device=x.device)

    k = torch.arange(n, dtype=torch.float32, device=x.device)
    w = torch.exp(-1j * 2.0 * torch.pi * k / n).to(torch.complex64)

    x_c = x.to(torch.complex64)
    return (2.0 / n) * torch.sum(x_c * w, dim=-1)

def compute_classical_distance_real_torch(
    z_app: torch.Tensor,
    r1: torch.Tensor,
    x1: torch.Tensor,
    line_len_km: torch.Tensor,
) -> torch.Tensor:
    z1 = torch.complex(r1, x1)
    eps = 1e-12
    d = torch.real(z_app / (z1 + eps)) * line_len_km
    zero = torch.zeros_like(d)
    return torch.minimum(torch.maximum(d, zero), line_len_km)



def classical_alpha_torch(device: torch.device) -> torch.Tensor:
    return torch.tensor(
        complex(-0.5, np.sqrt(3.0) / 2.0),
        dtype=torch.complex64,
        device=device,
    )

def compute_loop_phasors_for_case(
    Va: complex,
    Vb: complex,
    Vc: complex,
    Ia: complex,
    Ib: complex,
    Ic: complex,
    r1: float,
    x1: float,
    r0: float,
    x0: float,
    case: str,
) -> tuple[complex, complex, str]:
    """
    Build voltage/current loop phasors for the selected fault case.

    Returns:
        V_loop, I_loop, reason
    """
    eps = 1e-9
    case = str(case).lower().strip()

    i0_res = Ia + Ib + Ic
    k0 = k0_from_line(r1, x1, r0, x0)/3.0

    if case == "3ph":
        V_loop = symm_pos_seq(Va, Vb, Vc)
        I_loop = symm_pos_seq(Ia, Ib, Ic)

    elif case == "slg_a":
        V_loop = Va
        I_loop = Ia + k0 * i0_res

    elif case == "slg_b":
        V_loop = Vb
        I_loop = Ib + k0 * i0_res

    elif case == "slg_c":
        V_loop = Vc
        I_loop = Ic + k0 * i0_res

    elif case == "ll_ab":
        V_loop = Va - Vb
        I_loop = Ia - Ib

    elif case == "ll_bc":
        V_loop = Vb - Vc
        I_loop = Ib - Ic

    elif case == "ll_ca":
        V_loop = Vc - Va
        I_loop = Ic - Ia

    elif case == "llg_ab":
        V_loop = Va - Vb
        I_loop = Ia - Ib

    elif case == "llg_bc":
        V_loop = Vb - Vc
        I_loop = Ib - Ic

    elif case == "llg_ca":
        V_loop = Vc - Va
        I_loop = Ic - Ia

    else:
        return np.nan + 1j * np.nan, np.nan + 1j * np.nan, "unknown_case"

    if abs(I_loop) < eps:
        return V_loop, I_loop, "loop_current_too_small"

    return V_loop, I_loop, "ok"


def compute_modified_takagi_tf_only_from_window(
    x_raw: np.ndarray,
    fs: float,
    f_nom: float,
    r1: float,
    x1: float,
    r0: float,
    x0: float,
    case: str,
    dt_start: float,
    onset_idx_from_dt_start_fn,
    Z0_src_near: complex,
    Z0_src_far: complex,
    m_for_angle_pct: float,
    angle_sign: float = 1.0,
    clip_output=True,
) -> tuple[float, str]:
    """
    Modified Takagi-style SLG operator using graph-derived transformer/source
    zero-sequence approximation.

    This is intended for SLG ablation only.

    Uses:
        I_ref = 3I0 * exp(-jT)

    where T is derived from the zero-sequence network angle.

    Note:
        Z0_src_near / Z0_src_far currently come from the transformer-only
        fallback unless external-grid source parameters are available.
    """
    case = str(case).lower().strip()

    if case not in {"slg_a", "slg_b", "slg_c"}:
        return float("nan"), "not_slg"

    if x_raw.shape[1] not in {6, 12}:
        return float("nan"), f"unexpected_channel_count_{x_raw.shape[1]}"

    if x_raw.shape[1] == 12:
        x_raw = x_raw[:, :6]

    spc = int(np.rint(fs / f_nom))
    if spc <= 1:
        return float("nan"), "invalid_spc"

    onset_idx = onset_idx_from_dt_start_fn(dt_start, fs)

    post_start = onset_idx
    post_end = onset_idx + spc

    if post_end > x_raw.shape[0]:
        return float("nan"), "post_window_out_of_bounds"

    Va = dft_phasor_1cycle(x_raw[post_start:post_end, 0])
    Vb = dft_phasor_1cycle(x_raw[post_start:post_end, 1])
    Vc = dft_phasor_1cycle(x_raw[post_start:post_end, 2])
    Ia = dft_phasor_1cycle(x_raw[post_start:post_end, 3])
    Ib = dft_phasor_1cycle(x_raw[post_start:post_end, 4])
    Ic = dft_phasor_1cycle(x_raw[post_start:post_end, 5])

    if case == "slg_a":
        V_phase = Va
        I_phase = Ia
    elif case == "slg_b":
        V_phase = Vb
        I_phase = Ib
    elif case == "slg_c":
        V_phase = Vc
        I_phase = Ic
    else:
        return float("nan"), "not_slg"

    I0_res = Ia + Ib + Ic  # this is 3I0

    z1 = complex(r1, x1)
    z0 = complex(r0, x0)

    if abs(z1) < 1e-12:
        return float("nan"), "invalid_z1"

    if abs(I0_res) < 1e-9:
        return float("nan"), "zero_sequence_current_too_small"

    # Keep your current working k0 convention.
    k0 = k0_from_line(r1, x1, r0, x0)/3.0
    I_loop = I_phase + k0 * I0_res

    if abs(I_loop) < 1e-9:
        return float("nan"), "loop_current_too_small"

    if not np.isfinite(m_for_angle_pct):
        return float("nan"), "invalid_m_for_angle"

    m = float(np.clip(m_for_angle_pct / 100.0, 0.0, 1.0))

    # Zero-sequence current distribution approximation.
    Z0_far_path = Z0_src_far + (1.0 - m) * z0
    Z0_total = Z0_src_near + z0 + Z0_src_far

    if abs(Z0_total) < 1e-12:
        return float("nan"), "invalid_z0_total"

    current_distribution = Z0_far_path / Z0_total
    T = angle_sign * np.angle(current_distribution)

    I_ref = I0_res * np.exp(-1j * T)

    numerator = np.imag(V_phase * np.conj(I_ref))
    denominator = np.imag(z1 * I_loop * np.conj(I_ref))

    if abs(denominator) < 1e-9:
        return float("nan"), "mod_takagi_denominator_too_small"

    d_pu = numerator / denominator
    d_raw_pct = 100.0 * (d_pu)

    if not np.isfinite(d_raw_pct):
        return np.nan, "nonfinite_distance"

    if clip_output:
        return float(np.clip(d_raw_pct, 0.0, 100.0)), "ok"

    return float(d_raw_pct), "ok"


def compute_takagi_distance_from_window(
    x_raw: np.ndarray,
    fs: float,
    f_nom: float,
    r1: float,
    x1: float,
    r0: float,
    x0: float,
    case: str,
    dt_start: float,
    onset_idx_from_dt_start_fn,
) -> tuple[float, str]:
    """
    Compute Takagi-style distance estimate in percent of line length.

    Uses:
        I_sup = I_loop_post - I_loop_pre

        m = imag(V_loop_post * conj(I_sup))
            / imag(Z1 * I_loop_post * conj(I_sup))

    Here Z1 is used consistently with the existing operator convention.
    Output is clipped to [0, 100].
    """
    if x_raw.shape[1] not in {6, 12}:
        return float("nan"), f"unexpected_channel_count_{x_raw.shape[1]}"

    if x_raw.shape[1] == 12:
        x_raw = x_raw[:, :6]

    spc = int(np.rint(fs / f_nom))
    if spc <= 1:
        return float("nan"), "invalid_spc"

    onset_idx = onset_idx_from_dt_start_fn(dt_start, fs)

    pre_start = onset_idx - spc
    pre_end = onset_idx
    post_start = onset_idx
    post_end = onset_idx + spc

    if pre_start < 0:
        return float("nan"), "pre_window_out_of_bounds"
    if post_end > x_raw.shape[0]:
        return float("nan"), "post_window_out_of_bounds"

    # Pre-fault phasors
    Va_pre = dft_phasor_1cycle(x_raw[pre_start:pre_end, 0])
    Vb_pre = dft_phasor_1cycle(x_raw[pre_start:pre_end, 1])
    Vc_pre = dft_phasor_1cycle(x_raw[pre_start:pre_end, 2])
    Ia_pre = dft_phasor_1cycle(x_raw[pre_start:pre_end, 3])
    Ib_pre = dft_phasor_1cycle(x_raw[pre_start:pre_end, 4])
    Ic_pre = dft_phasor_1cycle(x_raw[pre_start:pre_end, 5])

    # Post-fault phasors
    Va_po = dft_phasor_1cycle(x_raw[post_start:post_end, 0])
    Vb_po = dft_phasor_1cycle(x_raw[post_start:post_end, 1])
    Vc_po = dft_phasor_1cycle(x_raw[post_start:post_end, 2])
    Ia_po = dft_phasor_1cycle(x_raw[post_start:post_end, 3])
    Ib_po = dft_phasor_1cycle(x_raw[post_start:post_end, 4])
    Ic_po = dft_phasor_1cycle(x_raw[post_start:post_end, 5])

    V_loop_po, I_loop_po, reason_po = compute_loop_phasors_for_case(
        Va=Va_po,
        Vb=Vb_po,
        Vc=Vc_po,
        Ia=Ia_po,
        Ib=Ib_po,
        Ic=Ic_po,
        r1=r1,
        x1=x1,
        r0=r0,
        x0=x0,
        case=case,
    )

    _, I_loop_pre, reason_pre = compute_loop_phasors_for_case(
        Va=Va_pre,
        Vb=Vb_pre,
        Vc=Vc_pre,
        Ia=Ia_pre,
        Ib=Ib_pre,
        Ic=Ic_pre,
        r1=r1,
        x1=x1,
        r0=r0,
        x0=x0,
        case=case,
    )

    if reason_po != "ok":
        return float("nan"), f"post_{reason_po}"
    if reason_pre != "ok":
        return float("nan"), f"pre_{reason_pre}"

    I_sup = I_loop_po - I_loop_pre

    if abs(I_sup) < 1e-9:
        return float("nan"), "superposition_current_too_small"

    z1 = complex(r1, x1)

    numerator = np.imag(V_loop_po * np.conj(I_sup))
    denominator = np.imag(z1 * I_loop_po * np.conj(I_sup))

    if abs(denominator) < 1e-9:
        return float("nan"), "takagi_denominator_too_small"

    d_pu = numerator / denominator
    d_pct = 100.0 * float(d_pu)

    if not np.isfinite(d_pct):
        return float("nan"), "takagi_not_finite"

    d_pct = float(np.clip(d_pct, 0.0, 100.0))
    return d_pct, "ok"

def compute_distance_with_learned_params(
    batch: dict,
    delta_alpha_re: torch.Tensor,
    delta_alpha_im: torch.Tensor,
    delta_k0_re: torch.Tensor,
    delta_k0_im: torch.Tensor,
    fs: float,
    f_nom: float = 50.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      d_pred_norm      : learned-operator normalized distance [0,1]
      d_classical_norm : classical normalized distance [0,1]
      alpha_learned_all
      k0_learned_all
    """
    x_seq = batch["x_seq"]   # (B, T, 6)
    case_idx = batch["case_idx"]
    r1 = batch["r1"]
    x1 = batch["x1"]
    r0 = batch["r0"]
    x0 = batch["x0"]
    line_len_km = batch["line_len_km"]
    dt_start = batch["dt_start"]

    B, T, F = x_seq.shape
    if F != 6:
        raise ValueError(f"Expected 6 VI channels, got {F}")

    spc = int(np.rint(fs / f_nom))
    onset_idx = torch.round((-dt_start) * fs).long()

    d_pred_list = []
    d_classical_list = []
    alpha_learned_list = []
    k0_learned_list = []

    eps = 1e-9
    alpha_classical = classical_alpha_torch(x_seq.device)

    idx_to_case = {v: k for k, v in ALL_CASE_TO_IDX.items()}

    for b in range(B):
        t0 = int(onset_idx[b].item())
        post_start = t0
        post_end = t0 + spc

        if post_start < 0 or post_end > T:
            zero_f = torch.tensor(0.0, dtype=torch.float32, device=x_seq.device)
            zero_c = torch.tensor(0.0 + 0.0j, dtype=torch.complex64, device=x_seq.device)
            d_pred_list.append(zero_f)
            d_classical_list.append(zero_f)
            alpha_learned_list.append(alpha_classical)
            k0_learned_list.append(zero_c)
            continue

        win = x_seq[b, post_start:post_end, :]

        Va = dft_phasor_1cycle_torch(win[:, 0])
        Vb = dft_phasor_1cycle_torch(win[:, 1])
        Vc = dft_phasor_1cycle_torch(win[:, 2])
        Ia = dft_phasor_1cycle_torch(win[:, 3])
        Ib = dft_phasor_1cycle_torch(win[:, 4])
        Ic = dft_phasor_1cycle_torch(win[:, 5])

        I0 = (Ia + Ib + Ic) / 3.0

        k0_classical = k0_from_line_torch(
            r1[b].unsqueeze(0),
            x1[b].unsqueeze(0),
            r0[b].unsqueeze(0),
            x0[b].unsqueeze(0),
        ).squeeze(0)

        alpha_learned = alpha_classical + torch.complex(
            delta_alpha_re[b], delta_alpha_im[b]
        )
        k0_learned = k0_classical + torch.complex(
            delta_k0_re[b], delta_k0_im[b]
        )

        case_name = idx_to_case[int(case_idx[b].item())]

        # Classical branch
        # Classical branch
        # Classical branch
        if case_name == "3ph":
            v1_c = (Va + alpha_classical * Vb + (alpha_classical ** 2) * Vc) / 3.0
            i1_c = (Ia + alpha_classical * Ib + (alpha_classical ** 2) * Ic) / 3.0
            z_app_classical = v1_c / (i1_c + eps)

            v1_l = (Va + alpha_learned * Vb + (alpha_learned ** 2) * Vc) / 3.0
            i1_l = (Ia + alpha_learned * Ib + (alpha_learned ** 2) * Ic) / 3.0
            z_app_learned = v1_l / (i1_l + eps)

        elif case_name == "slg_a":
            z_app_classical = Va / (Ia + k0_classical * I0 + eps)
            z_app_learned = Va / (Ia + k0_learned * I0 + eps)

        elif case_name == "slg_b":
            z_app_classical = Vb / (Ib + k0_classical * I0 + eps)
            z_app_learned = Vb / (Ib + k0_learned * I0 + eps)

        elif case_name == "slg_c":
            z_app_classical = Vc / (Ic + k0_classical * I0 + eps)
            z_app_learned = Vc / (Ic + k0_learned * I0 + eps)

        elif case_name == "ll_ab":
            z_app_classical = (Va - Vb) / ((Ia - Ib) + eps)
            z_app_learned = z_app_classical

        elif case_name == "ll_bc":
            z_app_classical = (Vb - Vc) / ((Ib - Ic) + eps)
            z_app_learned = z_app_classical

        elif case_name == "ll_ca":
            z_app_classical = (Vc - Va) / ((Ic - Ia) + eps)
            z_app_learned = z_app_classical

        elif case_name == "llg_ab":
            z_app_classical = (Va - Vb) / ((Ia - Ib) + eps)
            z_app_learned = z_app_classical

        elif case_name == "llg_bc":
            z_app_classical = (Vb - Vc) / ((Ib - Ic) + eps)
            z_app_learned = z_app_classical

        elif case_name == "llg_ca":
            z_app_classical = (Vc - Va) / ((Ic - Ia) + eps)
            z_app_learned = z_app_classical

        else:
            raise ValueError(f"Unsupported case: {case_name}")

        d_km_classical = compute_classical_distance_real_torch(
            z_app=z_app_classical,
            r1=r1[b],
            x1=x1[b],
            line_len_km=line_len_km[b],
        )
        d_km_learned = compute_classical_distance_real_torch(
            z_app=z_app_learned,
            r1=r1[b],
            x1=x1[b],
            line_len_km=line_len_km[b],
        )

        d_classical_norm = d_km_classical / (line_len_km[b] + eps)
        d_pred_norm = d_km_learned / (line_len_km[b] + eps)

        d_pred_list.append(d_pred_norm)
        d_classical_list.append(d_classical_norm)
        alpha_learned_list.append(alpha_learned)
        k0_learned_list.append(k0_learned)

    d_pred = torch.stack(d_pred_list, dim=0)
    d_classical = torch.stack(d_classical_list, dim=0)
    alpha_learned_all = torch.stack(alpha_learned_list, dim=0)
    k0_learned_all = torch.stack(k0_learned_list, dim=0)

    return d_pred, d_classical, alpha_learned_all, k0_learned_all
