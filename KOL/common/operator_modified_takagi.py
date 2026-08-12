from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from KOL.common.physics_core import compute_modified_takagi_tf_only_from_window
from KOL.common.windowing import onset_idx_from_dt_start


def add_empty_modified_takagi_columns(row_out: dict[str, Any]) -> None:
    row_out["d_mod_takagi_tf_only_p_local_pct"] = np.nan
    row_out["d_mod_takagi_tf_only_m_local_pct"] = np.nan
    row_out["d_mod_takagi_tf_only_p_remote_pct"] = np.nan
    row_out["d_mod_takagi_tf_only_m_remote_pct"] = np.nan
    row_out["d_mod_takagi_tf_only_p_remote_flipped_pct"] = np.nan
    row_out["d_mod_takagi_tf_only_m_remote_flipped_pct"] = np.nan
    row_out["d_mod_takagi_tf_only_p_both_mean_pct"] = np.nan
    row_out["d_mod_takagi_tf_only_m_both_mean_pct"] = np.nan
    row_out["mod_takagi_tf_only_p_reason_local"] = "not_slg"
    row_out["mod_takagi_tf_only_m_reason_local"] = "not_slg"
    row_out["mod_takagi_tf_only_p_reason_remote"] = "not_slg"
    row_out["mod_takagi_tf_only_m_reason_remote"] = "not_slg"
    row_out["mod_takagi_m_seed_local_pct"] = np.nan
    row_out["mod_takagi_m_seed_remote_pct"] = np.nan
    row_out["mod_takagi_m_seed_source_local"] = "not_slg"
    row_out["mod_takagi_m_seed_source_remote"] = "not_slg"


def _valid_pct_seed(value: Any) -> float | None:
    """Return a finite fault-distance seed clipped to [0, 100], else None."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(value):
        return None

    return float(np.clip(value, 0.0, 100.0))


def select_modified_takagi_seed_pct(
    feat: dict[str, Any],
) -> tuple[float, str]:
    """
    Select an inference-time seed for m in the modified-Takagi angle.

    Priority:
      1. Same-side standard Takagi estimate.
      2. Same-side classical impedance estimate.
      3. Fixed midpoint fallback only when neither is valid.

    No label or true fault location is used.
    """
    m_takagi = _valid_pct_seed(feat.get("d_takagi_pct"))
    if m_takagi is not None:
        return m_takagi, "standard_takagi_same_side"

    m_phys = _valid_pct_seed(feat.get("d_phys_real_pct"))
    if m_phys is not None:
        return m_phys, "zapp_real_same_side_fallback"

    return 50.0, "midpoint_no_valid_measurement_seed"



def add_modified_takagi_columns(
    *,
    row_out: dict[str, Any],
    row: pd.Series,
    x_vi_local: np.ndarray,
    x_vi_remote: np.ndarray,
    fs: float,
    f_nom: float,
    r1: float,
    x1: float,
    r0: float,
    x0: float,
    case: str,
    m_local_seed_pct: float,
    m_remote_seed_pct: float,
    m_local_seed_source: str,
    m_remote_seed_source: str,
    Z0_src_local: complex,
    Z0_src_remote: complex,
) -> None:
    """Enrich an SLG export row with auditable modified-Takagi variants.

    Local and remote inputs are terminal-oriented ``(T, 6)`` voltage/current
    arrays. Each seed is a percentage from its own relay terminal and is clipped
    before use; seed values and sources are exported to prove that no true fault
    label entered the calculation. Near/far zero-sequence source impedances are
    swapped for the remote calculation. Remote distances are later flipped into
    local orientation for paired means. Formula failures remain represented by
    NaN values and reason codes rather than rejecting the already valid base row.

    Args:
        row_out: Mutable operator row receiving percentage and reason columns.
        row: Source label row; only timing/identity metadata is consumed.
        case: Fault case; the scientific operator is meaningful only for SLG.
        m_local_seed_pct: Inference-time local-terminal distance seed, percent.
        m_remote_seed_pct: Inference-time remote-terminal distance seed, percent.
    """
    # m must be available at inference time.
    # Local and remote seeds are both expressed from their respective relay sides.
    m_local = float(np.clip(m_local_seed_pct, 0.0, 100.0))
    m_remote = float(np.clip(m_remote_seed_pct, 0.0, 100.0))

    # Export the chosen seeds so we can audit the non-leaking implementation.
    row_out["mod_takagi_m_seed_local_pct"] = m_local
    row_out["mod_takagi_m_seed_remote_pct"] = m_remote
    row_out["mod_takagi_m_seed_source_local"] = m_local_seed_source
    row_out["mod_takagi_m_seed_source_remote"] = m_remote_seed_source
    clip_output=False,

    d_mod_p_local_raw, reason_mod_p_local = compute_modified_takagi_tf_only_from_window(
        x_raw=x_vi_local,
        fs=fs,
        f_nom=float(f_nom),
        r1=r1,
        x1=x1,
        r0=r0,
        x0=x0,
        case=case,
        dt_start=float(row["dt_start"]),
        onset_idx_from_dt_start_fn=onset_idx_from_dt_start,
        Z0_src_near=Z0_src_local,
        Z0_src_far=Z0_src_remote,
        m_for_angle_pct=m_local,
        angle_sign=1.0,
        clip_output=False,
    )

    d_mod_m_local, reason_mod_m_local = compute_modified_takagi_tf_only_from_window(
        x_raw=x_vi_local,
        fs=fs,
        f_nom=float(f_nom),
        r1=r1,
        x1=x1,
        r0=r0,
        x0=x0,
        case=case,
        dt_start=float(row["dt_start"]),
        onset_idx_from_dt_start_fn=onset_idx_from_dt_start,
        Z0_src_near=Z0_src_local,
        Z0_src_far=Z0_src_remote,
        m_for_angle_pct=m_local,
        angle_sign=-1.0,
    )

    d_mod_p_remote_raw, reason_mod_p_remote = compute_modified_takagi_tf_only_from_window(
        x_raw=x_vi_remote,
        fs=fs,
        f_nom=float(f_nom),
        r1=r1,
        x1=x1,
        r0=r0,
        x0=x0,
        case=case,
        dt_start=float(row["dt_start"]),
        onset_idx_from_dt_start_fn=onset_idx_from_dt_start,
        Z0_src_near=Z0_src_remote,
        Z0_src_far=Z0_src_local,
        m_for_angle_pct=m_remote,
        angle_sign=1.0,
        clip_output=False,
    )

    d_mod_m_remote, reason_mod_m_remote = compute_modified_takagi_tf_only_from_window(
        x_raw=x_vi_remote,
        fs=fs,
        f_nom=float(f_nom),
        r1=r1,
        x1=x1,
        r0=r0,
        x0=x0,
        case=case,
        dt_start=float(row["dt_start"]),
        onset_idx_from_dt_start_fn=onset_idx_from_dt_start,
        Z0_src_near=Z0_src_remote,
        Z0_src_far=Z0_src_local,
        m_for_angle_pct=m_remote,
        angle_sign=-1.0,
    )

    row_out["d_mod_takagi_tf_only_p_local_pct"] = float(
        np.clip(d_mod_p_local_raw, 0.0, 100.0)
    )
    row_out["d_mod_takagi_tf_only_m_local_pct"] = d_mod_m_local
    row_out["d_mod_takagi_tf_only_p_remote_pct"] = float(
        np.clip(d_mod_p_remote_raw, 0.0, 100.0)
    )
    row_out["d_mod_takagi_tf_only_m_remote_pct"] = d_mod_m_remote

    row_out["d_mod_takagi_tf_only_p_remote_flipped_pct"] = float(
        np.clip(100.0 - d_mod_p_remote_raw, 0.0, 100.0)
    )
    row_out["d_mod_takagi_tf_only_m_remote_flipped_pct"] = 100.0 - d_mod_m_remote

    p_both_mean_raw = 0.5 * (
        d_mod_p_local_raw + (100.0 - d_mod_p_remote_raw)
    )

    row_out["d_mod_takagi_tf_only_p_both_mean_pct"] = float(
        np.clip(p_both_mean_raw, 0.0, 100.0)
    )
    row_out["d_mod_takagi_tf_only_m_both_mean_pct"] = 0.5 * (
        d_mod_m_local + (100.0 - d_mod_m_remote)
    )

    row_out["mod_takagi_tf_only_p_reason_local"] = reason_mod_p_local
    row_out["mod_takagi_tf_only_m_reason_local"] = reason_mod_m_local
    row_out["mod_takagi_tf_only_p_reason_remote"] = reason_mod_p_remote
    row_out["mod_takagi_tf_only_m_reason_remote"] = reason_mod_m_remote
