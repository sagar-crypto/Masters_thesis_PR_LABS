from __future__ import annotations

import numpy as np
import pandas as pd
from psp_helper.config import MainConfig


def onset_idx_from_dt_start(dt_start: float, fs: float) -> int:
    return int(np.rint((-float(dt_start)) * float(fs)))


def select_one_window_per_sample(
    df: pd.DataFrame,
    X_eval: np.ndarray,
    fs: float,
    f_nom: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    if "sample_id" not in df.columns:
        raise ValueError("df must contain 'sample_id'")
    if "dt_start" not in df.columns:
        raise ValueError("df must contain 'dt_start'")
    if "status" not in df.columns:
        raise ValueError("df must contain 'status'")

    work = df.copy().reset_index(drop=True)
    work["_row_idx"] = np.arange(len(work))

    spc = int(np.rint(fs / f_nom))
    T = X_eval.shape[1]

    work = work.loc[
        work["status"].astype(str).str.lower() == "fault_start"
    ].copy()

    work["_onset_idx"] = np.rint((-work["dt_start"].astype(float)) * fs).astype(int)

    work["_valid_timing"] = (
        (work["_onset_idx"] >= spc) &
        (work["_onset_idx"] + spc <= T)
    )
    work = work.loc[work["_valid_timing"]].copy()

    target_idx = T // 2
    work["_timing_score"] = np.abs(work["_onset_idx"] - target_idx)

    sort_cols = ["sample_id", "_timing_score"]
    ascending = [True, True]
    if "window_idx" in work.columns:
        sort_cols.append("window_idx")
        ascending.append(True)

    work = work.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    selected = work.groupby("sample_id", as_index=False).first()

    row_idx = selected["_row_idx"].to_numpy(dtype=int)
    X_sel = X_eval[row_idx]

    selected = selected.drop(
        columns=["_row_idx", "_onset_idx", "_valid_timing", "_timing_score"],
        errors="ignore",
    )

    return selected.reset_index(drop=True), X_sel


def select_one_window_per_sample_for_kol(
    df: pd.DataFrame,
    X_used: np.ndarray,
    window_s: float,
    f_nom: float = 50.0,
) -> tuple[pd.DataFrame, np.ndarray]:
    fs = X_used.shape[1] / float(window_s)
    return select_one_window_per_sample(
        df=df,
        X_eval=X_used,
        fs=fs,
        f_nom=f_nom,
    )


def filter_fault_start_windows_only(
    df: pd.DataFrame,
    X_used: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    if "status" not in df.columns:
        raise ValueError("df must contain 'status'")

    mask = df["status"].astype(str).str.lower() == "fault_start"
    df_out = df.loc[mask].reset_index(drop=True)
    X_out = X_used[mask.to_numpy()]
    return df_out, X_out


def filter_to_single_line_if_enabled(
    labels_df_used: pd.DataFrame,
    X_used: np.ndarray,
    config: MainConfig,
    logger=None,
) -> tuple[pd.DataFrame, np.ndarray]:
    line_filter = getattr(config.training, "line_filter", None)

    available_lines = sorted(labels_df_used["y_fault_line"].astype(str).unique().tolist())
    if logger is not None:
        logger.info("Available y_fault_line values: %s", available_lines)

    if line_filter is None:
        if logger is not None:
            logger.info("No single-line filter applied.")
        return labels_df_used, X_used

    line_filter = str(line_filter)

    if "y_fault_line" not in labels_df_used.columns:
        raise ValueError("labels_df_used does not contain 'y_fault_line'.")

    mask = labels_df_used["y_fault_line"].astype(str) == line_filter
    n_keep = int(mask.sum())

    if n_keep == 0:
        raise ValueError(
            f"No rows found for line_filter='{line_filter}'. "
            f"Available lines: {available_lines}"
        )

    labels_df_line = labels_df_used.loc[mask].reset_index(drop=True)
    X_line = X_used[mask.to_numpy()]

    if logger is not None:
        logger.info(
            "Applied single-line filter '%s': kept %d/%d rows (%.2f%%)",
            line_filter,
            len(labels_df_line),
            len(labels_df_used),
            100.0 * len(labels_df_line) / max(1, len(labels_df_used)),
        )

    return labels_df_line, X_line


def filter_to_single_line(
    labels_df_used: pd.DataFrame,
    X_used: np.ndarray,
    line_filter: str,
    logger=None,
) -> tuple[pd.DataFrame, np.ndarray]:
    mask = labels_df_used["y_fault_line"].astype(str) == str(line_filter)

    if int(mask.sum()) == 0:
        raise ValueError(f"No rows found for line_filter='{line_filter}'")

    df_out = labels_df_used.loc[mask].reset_index(drop=True)
    X_out = X_used[mask.to_numpy()]

    if logger is not None:
        logger.info(
            "Filtered to line='%s': kept %d/%d rows (%.2f%%)",
            line_filter,
            len(df_out),
            len(labels_df_used),
            100.0 * len(df_out) / max(1, len(labels_df_used)),
        )

    return df_out, X_out


def get_kol_mode(config: MainConfig) -> str:
    return str(
        getattr(config.training, "kol_prediction_mode", "ground_only_mul")
    ).lower().strip()


def filter_fault_start_windows_only_with_timing(
    df: pd.DataFrame,
    X_used: np.ndarray,
    fs: float,
    f_nom: float = 50.0,
) -> tuple[pd.DataFrame, np.ndarray]:
    if "status" not in df.columns:
        raise ValueError("df must contain 'status'")
    if "dt_start" not in df.columns:
        raise ValueError("df must contain 'dt_start'")

    work = df.copy().reset_index(drop=True)
    work["_row_idx"] = np.arange(len(work))

    T = X_used.shape[1]
    spc = int(np.rint(fs / f_nom))

    work = work.loc[
        work["status"].astype(str).str.lower() == "fault_start"
    ].copy()

    work["_onset_idx"] = np.rint((-work["dt_start"].astype(float)) * fs).astype(int)

    work["_valid_timing"] = (
        (work["_onset_idx"] >= 0) &
        (work["_onset_idx"] + spc <= T)
    )
    work = work.loc[work["_valid_timing"]].copy()

    row_idx = work["_row_idx"].to_numpy(dtype=int)
    X_out = X_used[row_idx]

    work = work.drop(
        columns=["_row_idx", "_onset_idx", "_valid_timing"],
        errors="ignore",
    ).reset_index(drop=True)

    return work, X_out
