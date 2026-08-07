from __future__ import annotations

from collections import Counter
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from psp_helper.config import MainConfig

from config import GRAPH_PATH

from KOL.common.operator_audit import audit_case_and_formula_mapping
from KOL.common.cv_utils import audit_cohort
from KOL.common.operator_data_prep import (
    apply_operator_window_selection,
    load_and_filter_operator_data,
)
from KOL.common.operator_row_builders import (
    build_both_side_operator_row,
    build_single_side_operator_row,
    build_two_ended_posseq_operator_row
)
from KOL.common.takagi_graph_impedances import load_takagi_impedance_bank


def run_formula_audit(
    *,
    df: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    audit_df = audit_case_and_formula_mapping(
        df=df,
        feature_names=feature_names,
        max_print=40,
    )

    audit_path = Path(os.environ.get("KOL_OUTPUT_ROOT", "outputs/reproducibility_validation/hydra_v1")) / "run_kol_formula_audit.csv"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(audit_path, index=False)
    print(f"\nSaved audit CSV to: {audit_path}")

    return audit_df


def print_and_save_operator_features(
    *,
    rows: list[dict[str, Any]],
    reason_counts: Counter,
    topology: str,
    operator_side_mode: str,
    operator_window_mode: str,
    output_root: str | None = None,
) -> pd.DataFrame:
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

    preview_candidates = [
        "sample_id",
        "window_idx",
        "y_fault_line",
        "y_fault_location",
        "case",
        "d_two_ended_posseq_plus_pct",
        "two_ended_posseq_plus_reason",
        "d_phys_real_pct",
        "d_phys_abs_pct",
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

    preview_cols = [c for c in preview_candidates if c in feat_df.columns]

    print(feat_df[preview_cols].head(10).to_string(index=False))

    unique_lines = sorted(feat_df["y_fault_line"].astype(str).unique())

    line_tag = unique_lines[0] if len(unique_lines) == 1 else "all_lines"
    line_tag = line_tag.replace("/", "_").replace(" ", "_")

    experiment_tag = (
        f"{topology}_{line_tag}_{operator_side_mode}_{operator_window_mode}"
    )
    out_path = Path(output_root or os.environ.get("KOL_OUTPUT_ROOT", "outputs/reproducibility_validation/hydra_v1")) / f"kol_operator_features_{experiment_tag}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    feat_df.to_csv(out_path, index=False)
    print(f"\nSaved operator features to: {out_path}")

    return feat_df


def export_operator_features(config: MainConfig) -> pd.DataFrame:
    df, X_eval, meta, fs, f_nom, y_col, topology = load_and_filter_operator_data(
        config
    )

    df, X_eval, operator_window_mode = apply_operator_window_selection(
        df=df,
        X_eval=X_eval,
        config=config,
        fs=fs,
        f_nom=f_nom,
    )

    if bool(getattr(config.training, "canonical_hard_audit", False)):
        from sklearn.model_selection import GroupKFold

        groups = df["sample_id"].to_numpy()
        splits = list(GroupKFold(n_splits=int(config.training.n_splits)).split(df, groups=groups))
        audit_cohort(
            df,
            splits,
            expected_rows=int(config.training.canonical_expected_rows),
            expected_events=int(config.training.canonical_expected_events),
            expected_folds=int(config.training.n_splits),
            expected_windows=getattr(config.training, "canonical_window_indices", None),
        )

    feature_names = list(meta["feature_names"])

    operator_side_mode = str(
        getattr(config.training, "operator_side_mode", "default")
    ).lower().strip()

    # The final two-ended export must not run the historical
    # one-ended / Takagi formula audit.
    if operator_side_mode != "two_ended_posseq":
        run_formula_audit(
            df=df,
            feature_names=feature_names,
        )

    print(f"Operator side mode: {operator_side_mode}")

    rows: list[dict[str, Any]] = []
    reason_counts: Counter = Counter()
    takagi_imp_bank = None

    if topology == "hv_double_line_110kv" and operator_side_mode == "both":
        takagi_imp_bank = load_takagi_impedance_bank(GRAPH_PATH)

    for i in range(len(df)):
        row = cast(pd.Series, df.iloc[i])
        x_raw_full = np.asarray(X_eval[i], dtype=np.float32)

        if operator_side_mode in {"default", "opposite"}:
            row_out, reason = build_single_side_operator_row(
                row=row,
                x_raw_full=x_raw_full,
                feature_names=feature_names,
                topology=topology,
                fs=fs,
                f_nom=f_nom,
                y_col=y_col,
                operator_side_mode=operator_side_mode,
            )

        elif operator_side_mode == "both":
            row_out, reason = build_both_side_operator_row(
                row=row,
                x_raw_full=x_raw_full,
                feature_names=feature_names,
                topology=topology,
                fs=fs,
                f_nom=f_nom,
                y_col=y_col,
                takagi_imp_bank=takagi_imp_bank,
            )

        elif operator_side_mode == "two_ended_posseq":
            row_out, reason = build_two_ended_posseq_operator_row(
                row=row,
                x_raw_full=x_raw_full,
                feature_names=feature_names,
                topology=topology,
                fs=fs,
                f_nom=f_nom,
                y_col=y_col,
            )

        else:
            row_out = None
            reason = f"unsupported_side_mode_{operator_side_mode}"

        if row_out is None:
            reason_counts[str(reason)] += 1
            continue

        rows.append(row_out)

    return print_and_save_operator_features(
        rows=rows,
        reason_counts=reason_counts,
        topology=topology,
        operator_side_mode=operator_side_mode,
        operator_window_mode=operator_window_mode,
        output_root=str(getattr(config.training, "out_dir", "outputs/reproducibility_validation/hydra_v1")),
    )
