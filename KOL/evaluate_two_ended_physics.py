from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from KOL.common.cv_utils import build_cv_splits_stratified


def convert_target_to_percentage(
    y_raw: np.ndarray,
) -> tuple[np.ndarray, str]:
    """
    Convert y_fault_location to percentage points only when it appears
    to be stored as a fraction in [0, 1].

    Returns:
        y_pct
        descriptive scale note
    """
    if not np.isfinite(y_raw).all():
        raise ValueError("y_fault_location contains non-finite values.")

    max_abs_value = float(np.max(np.abs(y_raw)))

    if max_abs_value <= 1.0 + 1e-6:
        return y_raw * 100.0, "fraction_to_percentage_x100"

    return y_raw, "already_percentage_points"


def calculate_metrics(
    y_true_pct: np.ndarray,
    y_pred_pct: np.ndarray,
) -> dict[str, float | int]:
    """
    Calculate raw-physics error statistics in percentage points.

    No clipping is performed here. Out-of-range predictions are retained
    in the MAE/RMSE calculation because they are genuine operator outputs.
    """
    absolute_error = np.abs(y_pred_pct - y_true_pct)

    out_of_range_mask = (
        (y_pred_pct < 0.0)
        | (y_pred_pct > 100.0)
    )

    return {
        "n_valid": int(len(y_true_pct)),
        "mae_pp": float(np.mean(absolute_error)),
        "rmse_pp": float(
            np.sqrt(np.mean((y_pred_pct - y_true_pct) ** 2))
        ),
        "median_abs_error_pp": float(np.median(absolute_error)),
        "p90_abs_error_pp": float(
            np.quantile(absolute_error, 0.90)
        ),
        "p95_abs_error_pp": float(
            np.quantile(absolute_error, 0.95)
        ),
        "max_abs_error_pp": float(np.max(absolute_error)),
        "out_of_range_count": int(np.sum(out_of_range_mask)),
        "out_of_range_rate": float(np.mean(out_of_range_mask)),
    }


def evaluate_raw_two_ended_physics(
    *,
    operator_path: str | Path,
    output_dir: str | Path,
    line_filter: str | None,
    prediction_column: str,
    reason_column: str,
    n_splits: int,
    split_seed: int,
    cv_mode: str,
    cv_stratify_col: str,
) -> None:
    """
    Evaluate raw synchronized two-ended positive-sequence estimates.

    The operator file is expected to contain:

        sample_id
        window_idx
        y_fault_location
        y_fault_line
        d_two_ended_posseq_plus_pct
        two_ended_posseq_plus_reason
    """
    operator_path = Path(operator_path)
    output_dir = Path(output_dir)

    df = pd.read_csv(operator_path)

    required_columns = {
        "sample_id",
        "window_idx",
        "y_fault_location",
        "y_fault_line",
        prediction_column,
    }

    missing_columns = sorted(
        required_columns.difference(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns in '{operator_path}': "
            f"{missing_columns}"
        )

    if line_filter is not None:
        df = df.loc[
            df["y_fault_line"].astype(str) == str(line_filter)
        ].copy()

    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "No rows remain after applying the line filter."
        )

    duplicate_keys = int(
        df.duplicated(["sample_id", "window_idx"]).sum()
    )

    if duplicate_keys > 0:
        raise ValueError(
            "Duplicate (sample_id, window_idx) keys found in the "
            f"operator CSV: {duplicate_keys}"
        )

    y_raw = pd.to_numeric(
        df["y_fault_location"],
        errors="coerce",
    ).to_numpy(dtype=float)

    if not np.isfinite(y_raw).all():
        raise ValueError(
            "y_fault_location contains non-numeric or non-finite values."
        )

    y_true_pct, target_scale_note = convert_target_to_percentage(
        y_raw
    )

    y_pred_pct = pd.to_numeric(
        df[prediction_column],
        errors="coerce",
    ).to_numpy(dtype=float)

    valid_prediction_mask = np.isfinite(y_pred_pct)

    n_total = int(len(df))
    n_valid = int(np.sum(valid_prediction_mask))
    n_invalid = int(np.sum(~valid_prediction_mask))

    if n_valid == 0:
        raise ValueError(
            "No finite raw physics predictions are available."
        )

    summary: dict[str, float | int | str] = {
        "operator_file": str(operator_path),
        "prediction_column": prediction_column,
        "line_filter": (
            "all_lines"
            if line_filter is None
            else str(line_filter)
        ),
        "n_selected_rows": n_total,
        "n_unique_events": int(df["sample_id"].nunique()),
        "n_valid_raw_rows": n_valid,
        "n_invalid_raw_rows": n_invalid,
        "invalid_raw_rate": float(n_invalid / n_total),
        "target_scale_note": target_scale_note,
    }

    summary.update(
        calculate_metrics(
            y_true_pct[valid_prediction_mask],
            y_pred_pct[valid_prediction_mask],
        )
    )

    # Use the same event-level grouped CV logic as KOL training.
    # sample_id keeps all windows belonging to the same event together.
    groups_np = df["sample_id"].to_numpy()

    splits = build_cv_splits_stratified(
        y_all=y_raw,
        groups_np=groups_np,
        task_type="regression",
        n_splits=int(n_splits),
        seed=int(split_seed),
        labels_df=df,
        cv_mode=str(cv_mode),
        stratify_col=str(cv_stratify_col),
    )

    fold_rows: list[dict[str, float | int]] = []

    for fold_index, (_train_idx, test_idx) in enumerate(splits):
        test_idx = np.asarray(test_idx, dtype=int)

        fold_valid_mask = valid_prediction_mask[test_idx]

        fold_row: dict[str, float | int] = {
            "fold": int(fold_index),
            "n_test_selected_rows": int(len(test_idx)),
            "n_test_unique_events": int(
                df.iloc[test_idx]["sample_id"].nunique()
            ),
            "n_test_valid_raw_rows": int(
                np.sum(fold_valid_mask)
            ),
            "n_test_invalid_raw_rows": int(
                np.sum(~fold_valid_mask)
            ),
        }

        if np.any(fold_valid_mask):
            valid_test_idx = test_idx[fold_valid_mask]

            fold_row.update(
                calculate_metrics(
                    y_true_pct[valid_test_idx],
                    y_pred_pct[valid_test_idx],
                )
            )

        else:
            fold_row.update(
                {
                    "n_valid": 0,
                    "mae_pp": np.nan,
                    "rmse_pp": np.nan,
                    "median_abs_error_pp": np.nan,
                    "p90_abs_error_pp": np.nan,
                    "p95_abs_error_pp": np.nan,
                    "max_abs_error_pp": np.nan,
                    "out_of_range_count": 0,
                    "out_of_range_rate": np.nan,
                }
            )

        fold_rows.append(fold_row)

    fold_metrics_df = pd.DataFrame(fold_rows)

    summary["mae_pp_fold_mean"] = float(
        fold_metrics_df["mae_pp"].mean()
    )

    summary["mae_pp_fold_std"] = float(
        fold_metrics_df["mae_pp"].std(ddof=1)
    )

    summary["rmse_pp_fold_mean"] = float(
        fold_metrics_df["rmse_pp"].mean()
    )

    summary["rmse_pp_fold_std"] = float(
        fold_metrics_df["rmse_pp"].std(ddof=1)
    )

    if reason_column in df.columns:
        reason_counts_df = (
            df[reason_column]
            .fillna("missing_reason")
            .value_counts(dropna=False)
            .rename_axis("reason")
            .reset_index(name="count")
        )
    else:
        reason_counts_df = pd.DataFrame(
            {
                "reason": ["reason_column_not_present"],
                "count": [n_total],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame([summary])

    summary_path = (
        output_dir
        / "two_ended_raw_physics_summary.csv"
    )

    fold_metrics_path = (
        output_dir
        / "two_ended_raw_physics_fold_metrics.csv"
    )

    reason_counts_path = (
        output_dir
        / "two_ended_raw_physics_reason_counts.csv"
    )

    summary_df.to_csv(summary_path, index=False)
    fold_metrics_df.to_csv(fold_metrics_path, index=False)
    reason_counts_df.to_csv(reason_counts_path, index=False)

    print("\nRaw two-ended physics summary:")
    print(summary_df.to_string(index=False))

    print("\nRaw operator reason counts:")
    print(reason_counts_df.to_string(index=False))

    print("\nSaved:")
    print(f"  {summary_path}")
    print(f"  {fold_metrics_path}")
    print(f"  {reason_counts_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the raw synchronized two-ended "
            "positive-sequence physics operator."
        )
    )

    parser.add_argument(
        "--operator-path",
        required=True,
        help="CSV produced by operator_side_mode=two_ended_posseq.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--line-filter",
        default=None,
    )

    parser.add_argument(
        "--prediction-column",
        default="d_two_ended_posseq_plus_pct",
    )

    parser.add_argument(
        "--reason-column",
        default="two_ended_posseq_plus_reason",
    )

    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--cv-mode",
        default="stratified_location",
    )

    parser.add_argument(
        "--cv-stratify-col",
        default="y_fault_location",
    )

    args = parser.parse_args()

    evaluate_raw_two_ended_physics(
        operator_path=args.operator_path,
        output_dir=args.output_dir,
        line_filter=args.line_filter,
        prediction_column=args.prediction_column,
        reason_column=args.reason_column,
        n_splits=args.n_splits,
        split_seed=args.split_seed,
        cv_mode=args.cv_mode,
        cv_stratify_col=args.cv_stratify_col,
    )


if __name__ == "__main__":
    main()
