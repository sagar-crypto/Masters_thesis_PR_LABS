from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from KOL.common.signal_ops import dft_phasor_1cycle, symm_pos_seq, symm_zero_seq, symm_neg_seq
from KOL.common.physics_core import compute_zapp_from_window, compute_takagi_distance_from_window
from psp_helper.config import MainConfig


def clip_pct_with_flags(raw_pct: float) -> tuple[float, int, int]:
    clipped = float(np.clip(raw_pct, 0.0, 100.0))
    is_low = int(raw_pct < 0.0)
    is_high = int(raw_pct > 100.0)
    return clipped, is_low, is_high


def edge_distance_score(pct: float) -> float:
    return float(min(abs(pct - 0.0), abs(100.0 - pct)))


def load_operator_inputs_if_enabled(
    config: MainConfig,
    labels_df_used: pd.DataFrame,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], list[str]]:
    use_ops = bool(getattr(config.training, "use_operator_features", False))
    if not use_ops:
        return None, None, []

    operator_path = getattr(config.training, "operator_features_path", None)
    if operator_path is None:
        raise ValueError(
            "use_operator_features=true but training.operator_features_path is not set."
        )

    if str(operator_path).endswith(".parquet"):
        ops_df = pd.read_parquet(operator_path)
    else:
        ops_df = pd.read_csv(operator_path)

    merge_keys = ["sample_id"]
    if "window_idx" in labels_df_used.columns and "window_idx" in ops_df.columns:
        merge_keys = ["sample_id", "window_idx"]

    merged = labels_df_used.reset_index(drop=True).merge(
        ops_df,
        on=merge_keys,
        how="left",
        suffixes=("", "_op"),
    )

    if len(merged) != len(labels_df_used):
        raise RuntimeError("Operator merge changed row count unexpectedly.")

    if "d_phys_real_pct" not in merged.columns:
        raise ValueError("Missing column 'd_phys_real_pct' in operator file.")

    d_phys_prior = merged["d_phys_real_pct"].astype(np.float32).to_numpy()

    if np.isnan(d_phys_prior).any():
        missing_mask = np.isnan(d_phys_prior)
        preview_cols = ["sample_id"]
        if "window_idx" in merged.columns:
            preview_cols.append("window_idx")
        missing_preview = merged.loc[missing_mask, preview_cols].head(10)
        raise ValueError(
            f"NaN values found in d_phys_real_pct after merge. "
            f"Missing rows: {int(missing_mask.sum())}/{len(merged)}. "
            f"Example missing keys:\n{missing_preview.to_string(index=False)}"
        )

    d_phys_prior = d_phys_prior / 100.0

    configured_cols = getattr(config.training, "operator_feature_cols", None)

    if configured_cols is None:
        candidate_cols = [
            "ratio_V0_V1",
            "ratio_V2_V1",
            "ratio_I0_I1",
            "ratio_I2_I1",
            "abs_Z0_app",
            "abs_Z2_app",
            "d_both_diff_real_pct",
            "d_local_real_pct",
            "d_remote_real_flipped_pct",
            "ratio_V0_V1_local",
            "ratio_V2_V1_local",
            "ratio_I0_I1_local",
            "ratio_I2_I1_local",
            "ratio_V0_V1_remote",
            "ratio_V2_V1_remote",
            "ratio_I0_I1_remote",
            "ratio_I2_I1_remote",
        ]
        operator_feature_cols = [c for c in candidate_cols if c in merged.columns]
    else:
        operator_feature_cols = [str(c) for c in configured_cols]
        missing_cols = [c for c in operator_feature_cols if c not in merged.columns]
        if missing_cols:
            raise ValueError(
                f"Configured operator_feature_cols missing in operator file: {missing_cols}"
            )

    if len(operator_feature_cols) == 0:
        return d_phys_prior, None, []

    op_features = merged[operator_feature_cols].astype(np.float32).to_numpy()

    if np.isnan(op_features).any():
        bad_rows = np.argwhere(np.isnan(op_features))
        raise ValueError(
            f"NaN values found in operator feature matrix. "
            f"First bad row/col: {bad_rows[0].tolist()} | cols={operator_feature_cols}"
        )

    return d_phys_prior, op_features, operator_feature_cols


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
    return 0.05 if (is_low == 1 or is_high == 1) else 1.0


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


def compute_single_side_operator_features(
    x_vi: np.ndarray,
    fs: float,
    f_nom: float,
    r1: float,
    x1: float,
    r0: float,
    x0: float,
    L_km: float,
    case: str,
    dt_start: float,
    onset_idx_from_dt_start_fn,
) -> dict:
    z_app, case_out, reason = compute_zapp_from_window(
        x_raw=x_vi,
        fs=fs,
        f_nom=f_nom,
        r1=r1,
        x1=x1,
        r0=r0,
        x0=x0,
        case=case,
        dt_start=dt_start,
        onset_idx_from_dt_start_fn=onset_idx_from_dt_start_fn,
    )

    if reason != "ok":
        return {
            "reason": reason,
            "case": case_out,
        }

    d_takagi_pct, takagi_reason = compute_takagi_distance_from_window(
        x_raw=x_vi,
        fs=fs,
        f_nom=f_nom,
        r1=r1,
        x1=x1,
        r0=r0,
        x0=x0,
        case=case,
        dt_start=dt_start,
        onset_idx_from_dt_start_fn=onset_idx_from_dt_start_fn,
    )
    z1 = complex(r1, x1)
    ratio_real = float(np.real(z_app / (z1 + 1e-12)))
    ratio_abs = float(abs(z_app) / (abs(z1) + 1e-12))

    d_phys_raw_real_km = float(ratio_real * L_km)
    d_phys_raw_real_pct = float(100.0 * d_phys_raw_real_km / L_km)
    d_phys_real_pct, is_clipped_low_real, is_clipped_high_real = clip_pct_with_flags(
        d_phys_raw_real_pct
    )
    d_phys_clipped_real_km = float((d_phys_real_pct / 100.0) * L_km)

    d_phys_raw_abs_km = float(ratio_abs * L_km)
    d_phys_raw_abs_pct = float(100.0 * d_phys_raw_abs_km / L_km)
    d_phys_abs_pct, is_clipped_low_abs, is_clipped_high_abs = clip_pct_with_flags(
        d_phys_raw_abs_pct
    )
    d_phys_clipped_abs_km = float((d_phys_abs_pct / 100.0) * L_km)

    spc = int(np.rint(fs / f_nom))
    onset_idx = onset_idx_from_dt_start_fn(dt_start, fs)
    post_start = onset_idx
    post_end = onset_idx + spc

    if post_start < 0 or post_end > x_vi.shape[0]:
        return {
            "reason": "post_window_out_of_bounds_for_seq",
            "case": case_out,
        }

    Va_po = dft_phasor_1cycle(x_vi[post_start:post_end, 0])
    Vb_po = dft_phasor_1cycle(x_vi[post_start:post_end, 1])
    Vc_po = dft_phasor_1cycle(x_vi[post_start:post_end, 2])
    Ia_po = dft_phasor_1cycle(x_vi[post_start:post_end, 3])
    Ib_po = dft_phasor_1cycle(x_vi[post_start:post_end, 4])
    Ic_po = dft_phasor_1cycle(x_vi[post_start:post_end, 5])

    eps_seq = 1e-12

    V0 = symm_zero_seq(Va_po, Vb_po, Vc_po)
    V1 = symm_pos_seq(Va_po, Vb_po, Vc_po)
    V2 = symm_neg_seq(Va_po, Vb_po, Vc_po)

    I0 = symm_zero_seq(Ia_po, Ib_po, Ic_po)
    I1 = symm_pos_seq(Ia_po, Ib_po, Ic_po)
    I2 = symm_neg_seq(Ia_po, Ib_po, Ic_po)

    Z0_app = V0 / (I0 + eps_seq)
    Z1_app = V1 / (I1 + eps_seq)
    Z2_app = V2 / (I2 + eps_seq)

    return {
        "reason": "ok",
        "case": case_out,
        "z_app": z_app,
        "z_app_real": float(np.real(z_app)),
        "z_app_imag": float(np.imag(z_app)),
        "ratio_real": ratio_real,
        "ratio_abs": ratio_abs,
        "d_phys_raw_real_km": d_phys_raw_real_km,
        "d_phys_clipped_real_km": d_phys_clipped_real_km,
        "d_phys_real_pct": d_phys_real_pct,
        "d_phys_raw_abs_km": d_phys_raw_abs_km,
        "d_phys_clipped_abs_km": d_phys_clipped_abs_km,
        "d_phys_abs_pct": d_phys_abs_pct,
        "abs_V0": float(np.abs(V0)),
        "abs_V1": float(np.abs(V1)),
        "abs_V2": float(np.abs(V2)),
        "abs_I0": float(np.abs(I0)),
        "abs_I1": float(np.abs(I1)),
        "abs_I2": float(np.abs(I2)),
        "ratio_V0_V1": float(np.abs(V0) / (np.abs(V1) + eps_seq)),
        "ratio_V2_V1": float(np.abs(V2) / (np.abs(V1) + eps_seq)),
        "ratio_I0_I1": float(np.abs(I0) / (np.abs(I1) + eps_seq)),
        "ratio_I2_I1": float(np.abs(I2) / (np.abs(I1) + eps_seq)),
        "abs_Z0_app": float(np.abs(Z0_app)),
        "abs_Z1_app": float(np.abs(Z1_app)),
        "abs_Z2_app": float(np.abs(Z2_app)),
        "d_phys_raw_real_pct": d_phys_raw_real_pct,
        "is_clipped_low_real": is_clipped_low_real,
        "is_clipped_high_real": is_clipped_high_real,
        "d_phys_raw_abs_pct": d_phys_raw_abs_pct,
        "is_clipped_low_abs": is_clipped_low_abs,
        "is_clipped_high_abs": is_clipped_high_abs,
        "d_takagi_pct": d_takagi_pct,
        "takagi_reason": takagi_reason,
        "takagi_valid": int(np.isfinite(d_takagi_pct)),
    }


def build_both_side_fusion_features(
    feat_local: dict,
    feat_remote: dict,
) -> dict:
    d_local_real_pct = feat_local["d_phys_real_pct"]
    d_remote_real_pct = feat_remote["d_phys_real_pct"]
    d_remote_real_flipped_pct = 100.0 - d_remote_real_pct

    d_local_raw_real_pct = feat_local["d_phys_raw_real_pct"]
    d_remote_raw_real_pct = feat_remote["d_phys_raw_real_pct"]
    d_remote_raw_real_flipped_pct = 100.0 - d_remote_raw_real_pct

    d_both_mean_real_pct = 0.5 * (d_local_real_pct + d_remote_real_flipped_pct)
    d_both_diff_real_pct = d_local_real_pct - d_remote_real_flipped_pct
    d_both_disagreement_real_pct = abs(d_both_diff_real_pct)
    d_both_min_real_pct = min(d_local_real_pct, d_remote_real_flipped_pct)
    d_both_max_real_pct = max(d_local_real_pct, d_remote_real_flipped_pct)

    d_both_edge_gated_real_pct = edge_gated_fusion(
        d_local_pct=d_local_real_pct,
        d_remote_flipped_pct=d_remote_real_flipped_pct,
        disagreement_threshold_pct=25.0,
    )

    w_local_real = confidence_weight_from_clip_flags(
        feat_local["is_clipped_low_real"],
        feat_local["is_clipped_high_real"],
    )
    w_remote_real = confidence_weight_from_clip_flags(
        feat_remote["is_clipped_low_real"],
        feat_remote["is_clipped_high_real"],
    )
    d_both_weighted_real_pct = weighted_fusion_from_confidence(
        d_local_pct=d_local_real_pct,
        d_remote_flipped_pct=d_remote_real_flipped_pct,
        w_local=w_local_real,
        w_remote=w_remote_real,
    )
    d_takagi_local_pct = feat_local.get("d_takagi_pct", np.nan)
    d_takagi_remote_pct = feat_remote.get("d_takagi_pct", np.nan)

    d_takagi_remote_flipped_pct = (
        100.0 - d_takagi_remote_pct
        if np.isfinite(d_takagi_remote_pct)
        else np.nan
    )

    if np.isfinite(d_takagi_local_pct) and np.isfinite(d_takagi_remote_flipped_pct):
        d_takagi_both_mean_pct = 0.5 * (
            d_takagi_local_pct + d_takagi_remote_flipped_pct
        )
        d_takagi_both_diff_pct = (
            d_takagi_local_pct - d_takagi_remote_flipped_pct
        )
        d_takagi_both_disagreement_pct = abs(d_takagi_both_diff_pct)
    else:
        d_takagi_both_mean_pct = np.nan
        d_takagi_both_diff_pct = np.nan
        d_takagi_both_disagreement_pct = np.nan

    d_local_abs_pct = feat_local["d_phys_abs_pct"]
    d_remote_abs_pct = feat_remote["d_phys_abs_pct"]
    d_remote_abs_flipped_pct = 100.0 - d_remote_abs_pct
    d_both_mean_abs_pct = 0.5 * (d_local_abs_pct + d_remote_abs_flipped_pct)
    d_both_diff_abs_pct = d_local_abs_pct - d_remote_abs_flipped_pct
    d_both_disagreement_abs_pct = abs(d_both_diff_abs_pct)

    return {
        "d_local_real_pct": d_local_real_pct,
        "d_remote_real_pct": d_remote_real_pct,
        "d_remote_real_flipped_pct": d_remote_real_flipped_pct,
        "d_local_raw_real_pct": d_local_raw_real_pct,
        "d_remote_raw_real_pct": d_remote_raw_real_pct,
        "d_remote_raw_real_flipped_pct": d_remote_raw_real_flipped_pct,
        "d_both_mean_real_pct": d_both_mean_real_pct,
        "d_both_diff_real_pct": d_both_diff_real_pct,
        "d_both_disagreement_real_pct": d_both_disagreement_real_pct,
        "d_both_min_real_pct": d_both_min_real_pct,
        "d_both_max_real_pct": d_both_max_real_pct,
        "d_both_edge_gated_real_pct": d_both_edge_gated_real_pct,
        "d_both_weighted_real_pct": d_both_weighted_real_pct,
        "w_local_real": w_local_real,
        "w_remote_real": w_remote_real,
        "d_local_abs_pct": d_local_abs_pct,
        "d_remote_abs_pct": d_remote_abs_pct,
        "d_remote_abs_flipped_pct": d_remote_abs_flipped_pct,
        "d_both_mean_abs_pct": d_both_mean_abs_pct,
        "d_both_diff_abs_pct": d_both_diff_abs_pct,
        "d_both_disagreement_abs_pct": d_both_disagreement_abs_pct,
        "is_local_clipped_low_real": feat_local["is_clipped_low_real"],
        "is_local_clipped_high_real": feat_local["is_clipped_high_real"],
        "is_remote_clipped_low_real": feat_remote["is_clipped_low_real"],
        "is_remote_clipped_high_real": feat_remote["is_clipped_high_real"],
        "d_phys_real_pct": d_both_weighted_real_pct,
        "d_phys_abs_pct": d_both_mean_abs_pct,
        "d_phys_real_strategy": "both_weighted_real",
        "d_takagi_local_pct": d_takagi_local_pct,
        "d_takagi_remote_pct": d_takagi_remote_pct,
        "d_takagi_remote_flipped_pct": d_takagi_remote_flipped_pct,
        "d_takagi_both_mean_pct": d_takagi_both_mean_pct,
        "d_takagi_both_diff_pct": d_takagi_both_diff_pct,
        "d_takagi_both_disagreement_pct": d_takagi_both_disagreement_pct,
        "takagi_valid_local": feat_local.get("takagi_valid", 0),
        "takagi_valid_remote": feat_remote.get("takagi_valid", 0),
        "takagi_reason_local": feat_local.get("takagi_reason", ""),
        "takagi_reason_remote": feat_remote.get("takagi_reason", ""),
    }
