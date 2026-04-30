from __future__ import annotations

from collections import Counter
from typing import cast

import hydra
import numpy as np
import pandas as pd
from psp_helper.config import MainConfig

from dl_psp.data.data_utils import load_windowed_dataset
from dl_psp.data.features import maybe_filter_features
from dl_psp.data.filters import (
    build_valid_row_indices_hv_double_line_90kv,
    build_valid_row_indices_hv_double_line_110kv,
)

from KOL.common.cases import (
    derive_fault_case_from_processed_labels,
    formula_name_for_case,
)
from KOL.common.line_utils import (
    attach_line_parameter_metadata,
    get_line_params_for_row,
)
from KOL.common.channel_mapping import (
    extract_line_vi_channels,
    get_line_vi_channel_names,
)
from KOL.common.windowing import (
    onset_idx_from_dt_start,
    select_one_window_per_sample,
    filter_fault_start_windows_only_with_timing
)
from KOL.common.operator_features import (
    compute_single_side_operator_features,
    build_both_side_fusion_features,
)


def audit_case_and_formula_mapping(
    df: pd.DataFrame,
    feature_names: list[str],
    max_print: int | None = 50,
) -> pd.DataFrame:
    rows = []

    for i in range(len(df)):
        row = cast(pd.Series, df.iloc[i])

        case = derive_fault_case_from_processed_labels(row)
        formula = formula_name_for_case(case)

        try:
            channel_names = get_line_vi_channel_names(
                feature_names=feature_names,
                fault_line=str(row["y_fault_line"]),
            )
            channel_ok = True
            channel_error = ""
        except Exception as e:
            channel_names = []
            channel_ok = False
            channel_error = str(e)

        a = int(row["y_phase_A"])
        b = int(row["y_phase_B"])
        c = int(row["y_phase_C"])
        grounded = int(row["y_is_grounded"])

        active = []
        if a == 1:
            active.append("a")
        if b == 1:
            active.append("b")
        if c == 1:
            active.append("c")

        if len(active) == 3:
            expected_case = "3ph"
        elif len(active) == 1 and grounded == 1:
            expected_case = f"slg_{active[0]}"
        elif len(active) == 2 and grounded == 0:
            pair = "".join(active)
            if pair == "ac":
                pair = "ca"
            expected_case = f"ll_{pair}"
        elif len(active) == 2 and grounded == 1:
            pair = "".join(active)
            if pair == "ac":
                pair = "ca"
            expected_case = f"llg_{pair}"
        else:
            expected_case = "invalid"

        case_ok = case == expected_case

        rows.append(
            {
                "row_idx": i,
                "sample_id": row["sample_id"],
                "status": row["status"],
                "fault_line": row["y_fault_line"],
                "fault_location_pct": float(row["y_fault_location"]),
                "dt_start": float(row["dt_start"]),
                "y_phase_A": a,
                "y_phase_B": b,
                "y_phase_C": c,
                "y_is_grounded": grounded,
                "derived_case": case,
                "expected_case": expected_case,
                "case_ok": case_ok,
                "formula": formula,
                "channel_ok": channel_ok,
                "channel_error": channel_error,
                "channels": " | ".join(channel_names),
            }
        )

    audit_df = pd.DataFrame(rows)

    print("\n===== CASE / FORMULA AUDIT =====")
    print(f"Total rows audited: {len(audit_df)}")
    print(f"Rows with case mismatch: {(~audit_df['case_ok']).sum()}")
    print(f"Rows with channel mapping error: {(~audit_df['channel_ok']).sum()}")

    if max_print is not None:
        print("\nFirst audited rows:")
        print(audit_df.head(max_print).to_string(index=False))

    return audit_df


@hydra.main(
    version_base=None,
    config_path="../third_party/dl_fault_repo/config",
    config_name="main-config.yaml",
)
def main(config: MainConfig) -> None:
    print("Loading processed windows...")
    X, labels_df, meta = load_windowed_dataset(config)

    include_groups = config.training.feature_groups_include
    materialize = config.training.materialize_feature_filters
    X_used, _feature_indices_for_ds = maybe_filter_features(
        X=X,
        meta=meta,
        include_groups=include_groups,
        materialize=materialize,
    )

    topology = str(config.dataset.topology)
    print(f"Running physics baseline for topology: {topology}")

    if topology == "hv_double_line_90kv":
        valid_row_idx = build_valid_row_indices_hv_double_line_90kv(
            labels_df, "y_fault_location"
        )
    elif topology == "hv_double_line_110kv":
        valid_row_idx = build_valid_row_indices_hv_double_line_110kv(
            labels_df, "y_fault_location"
        )
    else:
        raise ValueError(f"Unsupported topology for this baseline script: {topology}")

    if valid_row_idx is None:
        labels_df_used = labels_df.reset_index(drop=True)
        X_valid = X_used
    else:
        labels_df_used = labels_df.iloc[valid_row_idx].reset_index(drop=True)
        X_valid = X_used[valid_row_idx]

    print(f"Loaded X_valid shape: {X_valid.shape}")
    print(f"labels_df_used shape: {labels_df_used.shape}")

    line_filter = None
    loc_min = None
    loc_max = None
    f_nom = 50.0
    y_col = "y_fault_location"

    if topology == "hv_double_line_90kv":
        full_labels_path = "/home/vault/iwi5/iwi5305h/new_dataset_90kv/labels.csv"
    elif topology == "hv_double_line_110kv":
        full_labels_path = None
    else:
        raise ValueError(f"No full_labels_path configured for topology: {topology}")

    df = attach_line_parameter_metadata(
        labels_df_used=labels_df_used,
        full_labels_path=full_labels_path,
        topology=topology,
    )
    X_eval = X_valid

    print(f"Merged dataframe shape: {df.shape}")
    print("Unique y_fault_line values:")
    print(df["y_fault_line"].value_counts(dropna=False).to_string())

    if line_filter is not None:
        mask = df["y_fault_line"].astype(str) == str(line_filter)
        df = df.loc[mask].reset_index(drop=True)
        X_eval = X_eval[mask.to_numpy()]
        print(f"After line filter '{line_filter}': {len(df)} samples")

    y_pct = df[y_col].astype(float).to_numpy()

    if loc_min is not None:
        mask = y_pct >= loc_min
        df = df.loc[mask].reset_index(drop=True)
        X_eval = X_eval[mask]
        y_pct = y_pct[mask]

    if loc_max is not None:
        mask = y_pct <= loc_max
        df = df.loc[mask].reset_index(drop=True)
        X_eval = X_eval[mask]
        y_pct = y_pct[mask]

    T_full = X_eval.shape[1]
    window_s = float(config.window_extraction.window_length)
    fs = T_full / window_s
    print(f"Inferred fs: {fs:.3f} Hz from T={T_full}, window={window_s:.6f}s")

    df_typed: pd.DataFrame = cast(pd.DataFrame, df)

    operator_window_mode = str(
        getattr(config.training, "operator_window_mode", "single_fault_start")
    ).lower().strip()

    if operator_window_mode == "single_fault_start":
        df, X_eval = select_one_window_per_sample(
            df=df_typed,
            X_eval=X_eval,
            fs=fs,
            f_nom=f_nom,
        )
        print(f"Subset size after selecting one window per sample_id: {len(df)}")

    elif operator_window_mode == "all_fault_start":
        df, X_eval = filter_fault_start_windows_only_with_timing(
            df=df_typed,
            X_used=X_eval,
            fs=fs,
            f_nom=f_nom,
        )
        print(f"Subset size after keeping all valid fault_start windows: {len(df)}")

    else:
        raise ValueError(
            f"Unknown training.operator_window_mode='{operator_window_mode}'. "
            f"Supported: single_fault_start, all_fault_start"
        )

    print("Operator window mode:", operator_window_mode)
    print("Unique sample_id count after selection:", df["sample_id"].nunique())
    print("Status distribution after selection:")
    print(df["status"].value_counts(dropna=False).to_string())

    if "window_idx" in df.columns:
        print("\nWindow index distribution after selection:")
        print(df["window_idx"].value_counts().sort_index().to_string())

    spc = int(np.rint(fs / f_nom))
    onset_idx = np.rint((-df["dt_start"].astype(float)) * fs).astype(int)

    print("\ndt_start stats after selection:")
    print(df["dt_start"].describe().to_string())

    print("\nOnset index stats after selection:")
    print(pd.Series(onset_idx).describe().to_string())

    invalid_timing = ((onset_idx < spc) | (onset_idx + spc > X_eval.shape[1])).sum()
    print(f"\nRows violating timing after selection: {int(invalid_timing)}")

    feature_names = meta["feature_names"]

    audit_df = audit_case_and_formula_mapping(
        df=df,
        feature_names=feature_names,
        max_print=40,
    )

    audit_path = "/home/vault/iwi5/iwi5305h/run_kol_formula_audit.csv"
    audit_df.to_csv(audit_path, index=False)
    print(f"\nSaved audit CSV to: {audit_path}")

    operator_side_mode = str(
        getattr(config.training, "operator_side_mode", "default")
    ).lower().strip()

    print(f"Operator side mode: {operator_side_mode}")

    rows = []
    reason_counts = Counter()

    for i in range(len(df)):
        row = cast(pd.Series, df.iloc[i])
        x_raw_full = np.asarray(X_eval[i], dtype=np.float32)

        case = derive_fault_case_from_processed_labels(row)
        if case == "invalid":
            reason_counts["invalid_case_from_processed_labels"] += 1
            continue

        r1, x1, r0, x0, L_km = get_line_params_for_row(
            row=row,
            topology=topology,
        )

        if L_km <= 1e-12:
            reason_counts["invalid_line_length"] += 1
            continue

        if operator_side_mode in {"default", "opposite"}:
            try:
                x_vi, used_sides = extract_line_vi_channels(
                    x_raw=x_raw_full,
                    feature_names=feature_names,
                    fault_line=str(row["y_fault_line"]),
                    side_mode=operator_side_mode,
                )
            except Exception as e:
                reason_counts[f"channel_mapping_error: {type(e).__name__}"] += 1
                continue

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
                reason_counts[feat["reason"]] += 1
                continue

            row_out = {
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
                "z_app_real": feat["z_app_real"],
                "z_app_imag": feat["z_app_imag"],
                "ratio_real": feat["ratio_real"],
                "ratio_abs": feat["ratio_abs"],
                "d_phys_raw_real_km": feat["d_phys_raw_real_km"],
                "d_phys_clipped_real_km": feat["d_phys_clipped_real_km"],
                "d_phys_real_pct": feat["d_phys_real_pct"],
                "d_phys_raw_abs_km": feat["d_phys_raw_abs_km"],
                "d_phys_clipped_abs_km": feat["d_phys_clipped_abs_km"],
                "d_phys_abs_pct": feat["d_phys_abs_pct"],
                "abs_V0": feat["abs_V0"],
                "abs_V1": feat["abs_V1"],
                "abs_V2": feat["abs_V2"],
                "abs_I0": feat["abs_I0"],
                "abs_I1": feat["abs_I1"],
                "abs_I2": feat["abs_I2"],
                "ratio_V0_V1": feat["ratio_V0_V1"],
                "ratio_V2_V1": feat["ratio_V2_V1"],
                "ratio_I0_I1": feat["ratio_I0_I1"],
                "ratio_I2_I1": feat["ratio_I2_I1"],
                "abs_Z0_app": feat["abs_Z0_app"],
                "abs_Z1_app": feat["abs_Z1_app"],
                "abs_Z2_app": feat["abs_Z2_app"],
                "d_phys_raw_real_pct": feat["d_phys_raw_real_pct"],
                "is_clipped_low_real": feat["is_clipped_low_real"],
                "is_clipped_high_real": feat["is_clipped_high_real"],
                "d_phys_raw_abs_pct": feat["d_phys_raw_abs_pct"],
                "is_clipped_low_abs": feat["is_clipped_low_abs"],
                "is_clipped_high_abs": feat["is_clipped_high_abs"],
            }
            rows.append(row_out)

        elif operator_side_mode == "both":
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
                reason_counts[f"channel_mapping_error: {type(e).__name__}"] += 1
                continue

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
                reason_counts[f"local_{feat_local['reason']}"] += 1
                continue
            if feat_remote["reason"] != "ok":
                reason_counts[f"remote_{feat_remote['reason']}"] += 1
                continue

            fusion = build_both_side_fusion_features(feat_local, feat_remote)

            row_out = {
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
                "operator_side_mode": operator_side_mode,
                "used_sides": " | ".join(used_local + used_remote),

                "d_local_real_pct": fusion["d_local_real_pct"],
                "d_remote_real_pct": fusion["d_remote_real_pct"],
                "d_remote_real_flipped_pct": fusion["d_remote_real_flipped_pct"],
                "d_both_mean_real_pct": fusion["d_both_mean_real_pct"],
                "d_both_diff_real_pct": fusion["d_both_diff_real_pct"],

                "d_local_abs_pct": fusion["d_local_abs_pct"],
                "d_remote_abs_pct": fusion["d_remote_abs_pct"],
                "d_remote_abs_flipped_pct": fusion["d_remote_abs_flipped_pct"],
                "d_both_mean_abs_pct": fusion["d_both_mean_abs_pct"],
                "d_both_diff_abs_pct": fusion["d_both_diff_abs_pct"],

                "d_phys_real_pct": fusion["d_phys_real_pct"],
                "d_phys_abs_pct": fusion["d_phys_abs_pct"],
                "d_phys_real_strategy": "both_mean_real",

                "z_app_local_real": feat_local["z_app_real"],
                "z_app_local_imag": feat_local["z_app_imag"],
                "z_app_remote_real": feat_remote["z_app_real"],
                "z_app_remote_imag": feat_remote["z_app_imag"],

                "ratio_real_local": feat_local["ratio_real"],
                "ratio_real_remote": feat_remote["ratio_real"],
                "ratio_abs_local": feat_local["ratio_abs"],
                "ratio_abs_remote": feat_remote["ratio_abs"],

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

                "d_local_raw_real_pct": fusion["d_local_raw_real_pct"],
                "d_remote_raw_real_pct": fusion["d_remote_raw_real_pct"],
                "d_remote_raw_real_flipped_pct": fusion["d_remote_raw_real_flipped_pct"],

                "is_local_clipped_low_real": feat_local["is_clipped_low_real"],
                "is_local_clipped_high_real": feat_local["is_clipped_high_real"],
                "is_remote_clipped_low_real": feat_remote["is_clipped_low_real"],
                "is_remote_clipped_high_real": feat_remote["is_clipped_high_real"],

                "d_both_min_real_pct": fusion["d_both_min_real_pct"],
                "d_both_max_real_pct": fusion["d_both_max_real_pct"],
                "d_both_disagreement_real_pct": fusion["d_both_disagreement_real_pct"],
                "d_both_edge_gated_real_pct": fusion["d_both_edge_gated_real_pct"],
                "d_both_weighted_real_pct": fusion["d_both_weighted_real_pct"],

                "w_local_real": fusion["w_local_real"],
                "w_remote_real": fusion["w_remote_real"],

                "d_both_disagreement_abs_pct": fusion["d_both_disagreement_abs_pct"],
            }
            rows.append(row_out)

        else:
            reason_counts[f"unsupported_side_mode_{operator_side_mode}"] += 1
            continue

    feat_df = pd.DataFrame(rows)

    print("\n===== Operator feature export =====")
    print(f"Rows exported: {len(feat_df)}")

    if reason_counts:
        print("\nSkipped rows:")
        for k, v in reason_counts.most_common():
            print(f"{k}: {v}")

    if len(feat_df) == 0:
        raise RuntimeError("No operator features were produced.")

    print("\nOperator feature preview:")
    preview_cols = [
        "sample_id",
        "window_idx",
        "y_fault_line",
        "y_fault_location",
        "case",
        "d_phys_real_pct",
        "d_phys_abs_pct",
    ]

    extra_candidates = [
        "ratio_real",
        "ratio_abs",
        "ratio_V2_V1",
        "ratio_I2_I1",
        "d_both_mean_real_pct",
        "d_both_diff_real_pct",
        "d_both_disagreement_real_pct",
        "d_both_min_real_pct",
        "d_both_max_real_pct",
        "d_both_edge_gated_real_pct",
        "d_both_weighted_real_pct",
        "d_phys_real_strategy",
    ]
    preview_cols.extend([c for c in extra_candidates if c in feat_df.columns])

    print(feat_df[preview_cols].head(10).to_string(index=False))

    experiment_tag = f"{topology}_{operator_side_mode}_{operator_window_mode}_i0res_seq_bothfix"
    out_path = f"/home/vault/iwi5/iwi5305h/kol_operator_features_{experiment_tag}.csv"
    feat_df.to_csv(out_path, index=False)
    print(f"\nSaved operator features to: {out_path}")


if __name__ == "__main__":
    main()
