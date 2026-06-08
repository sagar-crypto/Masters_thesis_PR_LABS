from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from KOL.common.cases import derive_fault_case_from_processed_labels
from KOL.common.channel_mapping import extract_line_vi_channels
from KOL.common.line_utils import get_line_params_for_row
from KOL.common.operator_features import (
    build_both_side_fusion_features,
    compute_single_side_operator_features,
)
from KOL.common.operator_modified_takagi import (
    add_empty_modified_takagi_columns,
    add_modified_takagi_columns,
)
from KOL.common.windowing import onset_idx_from_dt_start


def build_single_side_operator_row(
    *,
    row: pd.Series,
    x_raw_full: np.ndarray,
    feature_names: list[str],
    topology: str,
    fs: float,
    f_nom: float,
    y_col: str,
    operator_side_mode: str,
) -> tuple[dict[str, Any] | None, str | None]:
    case = derive_fault_case_from_processed_labels(row)
    if case == "invalid":
        return None, "invalid_case_from_processed_labels"

    r1, x1, r0, x0, L_km = get_line_params_for_row(
        row=row,
        topology=topology,
    )

    if L_km <= 1e-12:
        return None, "invalid_line_length"

    try:
        x_vi, used_sides = extract_line_vi_channels(
            x_raw=x_raw_full,
            feature_names=feature_names,
            fault_line=str(row["y_fault_line"]),
            side_mode=operator_side_mode,
        )
    except Exception as e:
        return None, f"channel_mapping_error: {type(e).__name__}"

    feat = compute_single_side_operator_features(
        x_vi=x_vi,
        fs=fs,
        f_nom=float(f_nom),
        r1=r1,
        x1=x1,
        r0=r0,
        x0=x0,
        L_km=L_km,
        case=case,
        dt_start=float(row["dt_start"]),
        onset_idx_from_dt_start_fn=onset_idx_from_dt_start,
    )

    if feat["reason"] != "ok":
        return None, str(feat["reason"])

    row_out: dict[str, Any] = {
        "sample_id": row["sample_id"],
        "window_idx": row["window_idx"],
        "status": row["status"],
        "dt_start": float(row["dt_start"]),
        "y_fault_line": row["y_fault_line"],
        "y_fault_location": float(row[y_col]),
        "case": feat["case"],
        "r1": r1,
        "x1": x1,
        "r0": r0,
        "x0": x0,
        "line_len_km": L_km,
        "operator_side_mode": operator_side_mode,
        "used_sides": " | ".join(used_sides),
    }

    for k, v in feat.items():
        if k not in row_out:
            row_out[k] = v

    return row_out, None


def build_both_side_operator_row(
    *,
    row: pd.Series,
    x_raw_full: np.ndarray,
    feature_names: list[str],
    topology: str,
    fs: float,
    f_nom: float,
    y_col: str,
    takagi_imp_bank: dict[str, dict[str, complex]],
) -> tuple[dict[str, Any] | None, str | None]:
    case = derive_fault_case_from_processed_labels(row)
    if case == "invalid":
        return None, "invalid_case_from_processed_labels"

    r1, x1, r0, x0, L_km = get_line_params_for_row(
        row=row,
        topology=topology,
    )

    if L_km <= 1e-12:
        return None, "invalid_line_length"

    try:
        x_vi_local, used_local = extract_line_vi_channels(
            x_raw=x_raw_full,
            feature_names=feature_names,
            fault_line=str(row["y_fault_line"]),
            side_mode="default",
        )
        x_vi_remote, used_remote = extract_line_vi_channels(
            x_raw=x_raw_full,
            feature_names=feature_names,
            fault_line=str(row["y_fault_line"]),
            side_mode="opposite",
        )
    except Exception as e:
        return None, f"channel_mapping_error: {type(e).__name__}"

    feat_local = compute_single_side_operator_features(
        x_vi=x_vi_local,
        fs=fs,
        f_nom=float(f_nom),
        r1=r1,
        x1=x1,
        r0=r0,
        x0=x0,
        L_km=L_km,
        case=case,
        dt_start=float(row["dt_start"]),
        onset_idx_from_dt_start_fn=onset_idx_from_dt_start,
    )

    feat_remote = compute_single_side_operator_features(
        x_vi=x_vi_remote,
        fs=fs,
        f_nom=float(f_nom),
        r1=r1,
        x1=x1,
        r0=r0,
        x0=x0,
        L_km=L_km,
        case=case,
        dt_start=float(row["dt_start"]),
        onset_idx_from_dt_start_fn=onset_idx_from_dt_start,
    )

    if feat_local["reason"] != "ok":
        return None, f"local_{feat_local['reason']}"

    if feat_remote["reason"] != "ok":
        return None, f"remote_{feat_remote['reason']}"

    fusion = build_both_side_fusion_features(feat_local, feat_remote)

    fault_line = str(row["y_fault_line"])
    imp = takagi_imp_bank[fault_line]

    Z0_src_local = imp["Z0_src_local"]
    Z0_src_remote = imp["Z0_src_remote"]

    row_out: dict[str, Any] = {
        "sample_id": row["sample_id"],
        "window_idx": row["window_idx"],
        "status": row["status"],
        "dt_start": float(row["dt_start"]),
        "y_fault_line": row["y_fault_line"],
        "y_fault_location": float(row[y_col]),
        "case": feat_local["case"],
        "r1": r1,
        "x1": x1,
        "r0": r0,
        "x0": x0,
        "line_len_km": L_km,
        "operator_side_mode": "both",
        "used_sides": " | ".join(used_local + used_remote),

        # local-side details
        "z_app_local_real": feat_local["z_app_real"],
        "z_app_local_imag": feat_local["z_app_imag"],
        "ratio_real_local": feat_local["ratio_real"],
        "ratio_abs_local": feat_local["ratio_abs"],

        "abs_V0_local": feat_local["abs_V0"],
        "abs_V1_local": feat_local["abs_V1"],
        "abs_V2_local": feat_local["abs_V2"],
        "abs_I0_local": feat_local["abs_I0"],
        "abs_I1_local": feat_local["abs_I1"],
        "abs_I2_local": feat_local["abs_I2"],
        "ratio_V0_V1_local": feat_local["ratio_V0_V1"],
        "ratio_V2_V1_local": feat_local["ratio_V2_V1"],
        "ratio_I0_I1_local": feat_local["ratio_I0_I1"],
        "ratio_I2_I1_local": feat_local["ratio_I2_I1"],
        "abs_Z0_app_local": feat_local["abs_Z0_app"],
        "abs_Z1_app_local": feat_local["abs_Z1_app"],
        "abs_Z2_app_local": feat_local["abs_Z2_app"],

        # remote-side details
        "z_app_remote_real": feat_remote["z_app_real"],
        "z_app_remote_imag": feat_remote["z_app_imag"],
        "ratio_real_remote": feat_remote["ratio_real"],
        "ratio_abs_remote": feat_remote["ratio_abs"],

        "abs_V0_remote": feat_remote["abs_V0"],
        "abs_V1_remote": feat_remote["abs_V1"],
        "abs_V2_remote": feat_remote["abs_V2"],
        "abs_I0_remote": feat_remote["abs_I0"],
        "abs_I1_remote": feat_remote["abs_I1"],
        "abs_I2_remote": feat_remote["abs_I2"],
        "ratio_V0_V1_remote": feat_remote["ratio_V0_V1"],
        "ratio_V2_V1_remote": feat_remote["ratio_V2_V1"],
        "ratio_I0_I1_remote": feat_remote["ratio_I0_I1"],
        "ratio_I2_I1_remote": feat_remote["ratio_I2_I1"],
        "abs_Z0_app_remote": feat_remote["abs_Z0_app"],
        "abs_Z1_app_remote": feat_remote["abs_Z1_app"],
        "abs_Z2_app_remote": feat_remote["abs_Z2_app"],

        # optional single-side Takagi diagnostics
        "d_takagi_local_raw_pct": feat_local.get("d_takagi_pct", np.nan),
        "d_takagi_remote_raw_pct": feat_remote.get("d_takagi_pct", np.nan),
        "takagi_valid_local_raw": feat_local.get("takagi_valid", 0),
        "takagi_valid_remote_raw": feat_remote.get("takagi_valid", 0),
        "takagi_reason_local_raw": feat_local.get("takagi_reason", ""),
        "takagi_reason_remote_raw": feat_remote.get("takagi_reason", ""),
    }

    if case in {"slg_a", "slg_b", "slg_c"}:
        add_modified_takagi_columns(
            row_out=row_out,
            row=row,
            x_vi_local=x_vi_local,
            x_vi_remote=x_vi_remote,
            fs=fs,
            f_nom=f_nom,
            r1=r1,
            x1=x1,
            r0=r0,
            x0=x0,
            case=case,
            y_col=y_col,
            Z0_src_local=Z0_src_local,
            Z0_src_remote=Z0_src_remote,
        )
    else:
        add_empty_modified_takagi_columns(row_out)

    for k, v in fusion.items():
        if k not in row_out:
            row_out[k] = v

    return row_out, None
