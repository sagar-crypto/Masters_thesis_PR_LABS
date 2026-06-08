from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from psp_helper.config import MainConfig

from dl_psp.data.data_utils import load_windowed_dataset
from dl_psp.data.features import maybe_filter_features as apply_configured_feature_filter
from dl_psp.data.filters import (
    build_valid_row_indices_hv_double_line_90kv,
    build_valid_row_indices_hv_double_line_110kv,
)

from KOL.common.line_utils import attach_line_parameter_metadata
from KOL.common.operator_features import build_both_side_fusion_features
from KOL.common.windowing import (
    filter_fault_start_windows_only_with_timing,
    select_one_window_per_sample,
)


def load_and_filter_operator_data(
    config: MainConfig,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any], float, float, str, str]:
    print("Loading processed windows...")

    X, labels_df, meta = load_windowed_dataset(config)

    print(
        "DEBUG build_both_side_fusion_features loaded from:",
        build_both_side_fusion_features.__code__.co_filename,
    )

    include_groups = config.training.feature_groups_include
    materialize = config.training.materialize_feature_filters

    X_used, _feature_indices_for_ds = apply_configured_feature_filter(
        X=X,
        meta=meta,
        include_groups=include_groups,
        materialize=materialize,
    )

    topology = str(config.dataset.topology)
    print(f"Running physics baseline for topology: {topology}")

    y_col = "y_fault_location"
    f_nom = 50.0

    if topology == "hv_double_line_90kv":
        valid_row_idx = build_valid_row_indices_hv_double_line_90kv(
            labels_df,
            y_col,
        )
    elif topology == "hv_double_line_110kv":
        valid_row_idx = build_valid_row_indices_hv_double_line_110kv(
            labels_df,
            y_col,
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

    return df, X_eval, meta, fs, f_nom, y_col, topology


def apply_operator_window_selection(
    *,
    df: pd.DataFrame,
    X_eval: np.ndarray,
    config: MainConfig,
    fs: float,
    f_nom: float,
) -> tuple[pd.DataFrame, np.ndarray, str]:
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

    return df, X_eval, operator_window_mode
