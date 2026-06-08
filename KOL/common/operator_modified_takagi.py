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
    y_col: str,
    Z0_src_local: complex,
    Z0_src_remote: complex,
) -> None:
    m_local = float(row[y_col])
    m_remote = 100.0 - float(row[y_col])

    d_mod_p_local, reason_mod_p_local = compute_modified_takagi_tf_only_from_window(
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

    d_mod_p_remote, reason_mod_p_remote = compute_modified_takagi_tf_only_from_window(
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

    row_out["d_mod_takagi_tf_only_p_local_pct"] = d_mod_p_local
    row_out["d_mod_takagi_tf_only_m_local_pct"] = d_mod_m_local
    row_out["d_mod_takagi_tf_only_p_remote_pct"] = d_mod_p_remote
    row_out["d_mod_takagi_tf_only_m_remote_pct"] = d_mod_m_remote

    row_out["d_mod_takagi_tf_only_p_remote_flipped_pct"] = 100.0 - d_mod_p_remote
    row_out["d_mod_takagi_tf_only_m_remote_flipped_pct"] = 100.0 - d_mod_m_remote

    row_out["d_mod_takagi_tf_only_p_both_mean_pct"] = 0.5 * (
        d_mod_p_local + (100.0 - d_mod_p_remote)
    )
    row_out["d_mod_takagi_tf_only_m_both_mean_pct"] = 0.5 * (
        d_mod_m_local + (100.0 - d_mod_m_remote)
    )

    row_out["mod_takagi_tf_only_p_reason_local"] = reason_mod_p_local
    row_out["mod_takagi_tf_only_m_reason_local"] = reason_mod_m_local
    row_out["mod_takagi_tf_only_p_reason_remote"] = reason_mod_p_remote
    row_out["mod_takagi_tf_only_m_reason_remote"] = reason_mod_m_remote
