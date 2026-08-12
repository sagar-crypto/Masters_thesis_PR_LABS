from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from psp_helper.config import MainConfig

import dl_psp.data.labels as L
from dl_psp.data.data_utils import load_windowed_dataset
from dl_psp.data.features import maybe_filter_features as apply_configured_feature_filter
from dl_psp.data.filters import (
    build_valid_row_indices,
    build_valid_row_indices_hv_double_line_90kv,
    build_valid_row_indices_hv_double_line_110kv,
)

from KOL.common.windowing import (
    filter_fault_start_windows_only_with_timing,
    filter_to_single_line_if_enabled,
    select_one_window_per_sample_for_kol,
)


@dataclass
class PreparedTrainingData:
    X_used_filtered: np.ndarray
    labels_df_used: pd.DataFrame
    meta: dict[str, Any]
    feature_indices_for_ds: Any
    valid_row_idx: Any
    use_ops: bool
    kol_window_mode: str


def _apply_topology_valid_row_filter(
    *,
    X_used: np.ndarray,
    labels_df: pd.DataFrame,
    topology: str,
    target_label: str,
    logger,
) -> tuple[pd.DataFrame, np.ndarray, Any]:
    if topology == "hv_double_line_90kv":
        valid_row_idx = build_valid_row_indices_hv_double_line_90kv(
            labels_df, target_label
        )
        topology_msg = "hv_double_line_90kv"

    elif topology == "hv_double_line_110kv":
        valid_row_idx = build_valid_row_indices_hv_double_line_110kv(
            labels_df, target_label
        )
        topology_msg = "hv_double_line_110kv"

    else:
        valid_row_idx = build_valid_row_indices(labels_df, target_label=target_label)
        topology_msg = "generic target-specific"

    if valid_row_idx is None:
        labels_df_used = labels_df.reset_index(drop=True)
        X_used_filtered = X_used
        logger.info("No custom filtering applied for %s (valid_row_idx=None).", topology_msg)
    else:
        labels_df_used = labels_df.iloc[valid_row_idx].reset_index(drop=True)
        X_used_filtered = X_used[valid_row_idx]
        logger.info(
            "Applied custom filtering for %s: kept %d/%d rows (%.2f%%).",
            topology_msg,
            len(labels_df_used),
            len(labels_df),
            100.0 * (len(labels_df_used) / max(1, len(labels_df))),
        )

    return labels_df_used, X_used_filtered, valid_row_idx


def _apply_kol_window_filter_if_enabled(
    *,
    labels_df_used: pd.DataFrame,
    X_used_filtered: np.ndarray,
    config: MainConfig,
    use_ops: bool,
    kol_window_mode: str,
    logger,
) -> tuple[pd.DataFrame, np.ndarray]:
    if not use_ops:
        return labels_df_used, X_used_filtered

    if kol_window_mode == "single_fault_start":
        labels_df_used, X_used_filtered = select_one_window_per_sample_for_kol(
            df=labels_df_used,
            X_used=X_used_filtered,
            window_s=float(config.window_extraction.window_length),
            f_nom=50.0,
        )

    elif kol_window_mode == "all_fault_start":
        window_s = float(config.window_extraction.window_length)
        fs = X_used_filtered.shape[1] / window_s

        labels_df_used, X_used_filtered = filter_fault_start_windows_only_with_timing(
            df=labels_df_used,
            X_used=X_used_filtered,
            fs=fs,
            f_nom=50.0,
        )

        logger.info(
            "KOL window mode = all_fault_start with timing filter: kept %d rows | unique sample_id=%d",
            len(labels_df_used),
            int(labels_df_used[L.SAMPLE_ID].nunique()),
        )

        if "window_idx" in labels_df_used.columns:
            logger.info(
                "Window_idx distribution after timing filter:\n%s",
                labels_df_used["window_idx"].value_counts().sort_index().to_string(),
            )

    elif kol_window_mode == "all_valid":
        logger.info("KOL window mode = all_valid (no extra window filtering applied).")

    else:
        raise ValueError(
            f"Unknown training.kol_window_mode='{kol_window_mode}'. "
            f"Supported: single_fault_start, all_fault_start, all_valid"
        )

    return labels_df_used, X_used_filtered


def load_filtered_training_data(
    *,
    config: MainConfig,
    logger,
) -> PreparedTrainingData:
    """Load waveforms and establish the canonical training-row coordinate system.

    The private loader supplies ``X`` as ``(rows, timesteps, features)`` plus a
    row-aligned label frame. Feature-group selection is applied first, followed
    by topology/target validity, optional line selection, and the configured
    fault-start window policy. At 90/110 kV the public protocols subsequently
    expect effective tensors of ``384 x 48``/``576 x 48`` per retained row.

    Args:
        config: Private-schema configuration produced by the public adapter.
        logger: Logger receiving filter counts and window diagnostics.

    Returns:
        Filtered tensors and labels, source metadata, deferred feature indices,
        original valid-row indices, and the active operator/window modes. The
        returned labels and tensor share a fresh positional index; ``valid_row_idx``
        still refers to the pre-filter loader coordinates.
    """
    # Phase 1: load and apply canonical channel groups before row filtering.
    X, labels_df, meta = load_windowed_dataset(config)

    include_groups = config.training.feature_groups_include
    materialize = config.training.materialize_feature_filters

    X_used, feature_indices_for_ds = apply_configured_feature_filter(
        X=X,
        meta=meta,
        include_groups=include_groups,
        materialize=materialize,
    )

    target_label = str(config.training.target_label)
    topology = str(config.dataset.topology)

    # Phase 2: reject rows invalid for the topology/target scientific contract.
    labels_df_used, X_used_filtered, valid_row_idx = _apply_topology_valid_row_filter(
        X_used=X_used,
        labels_df=labels_df,
        topology=topology,
        target_label=target_label,
        logger=logger,
    )

    labels_df_used, X_used_filtered = filter_to_single_line_if_enabled(
        labels_df_used=labels_df_used,
        X_used=X_used_filtered,
        config=config,
        logger=logger,
    )

    use_ops = bool(
        getattr(
            config.training,
            "use_operator_features",
            False,
        )
    )

    kol_window_mode = str(
        getattr(
            config.training,
            "kol_window_mode",
            "single_fault_start",
        )
    )

    apply_window_filter_without_operator = bool(
        getattr(
            config.training,
            "apply_window_filter_without_operator",
            False,
        )
    )

    apply_window_filter = (
        use_ops
        or apply_window_filter_without_operator
    )

    # Phase 3: select the cohort's fault-start windows in filtered coordinates.
    labels_df_used, X_used_filtered = (
        _apply_kol_window_filter_if_enabled(
            labels_df_used=labels_df_used,
            X_used_filtered=X_used_filtered,
            config=config,
            use_ops=apply_window_filter,
            kol_window_mode=kol_window_mode,
            logger=logger,
        )
    )

    return PreparedTrainingData(
        X_used_filtered=X_used_filtered,
        labels_df_used=labels_df_used,
        meta=meta,
        feature_indices_for_ds=feature_indices_for_ds,
        valid_row_idx=valid_row_idx,
        use_ops=use_ops,
        kol_window_mode=kol_window_mode,
    )   
