from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


RAW_PRIOR_COL = "d_two_ended_posseq_plus_pct"
RAW_REASON_COL = "two_ended_posseq_plus_reason"
INPUT_PRIOR_COL = "d_two_ended_posseq_plus_input_pct"
FALLBACK_FLAG_COL = "two_ended_posseq_input_used_fallback"


def prepare_two_ended_prior_file(
    *,
    input_path: str | Path,
    output_path: str | Path,
    raw_prior_col: str = RAW_PRIOR_COL,
    raw_reason_col: str = RAW_REASON_COL,
    input_prior_col: str = INPUT_PRIOR_COL,
    fallback_pct: float = 50.0,
) -> pd.DataFrame:
    """
    Keep the raw two-ended physics result unchanged and create a separate,
    bounded prior column for neural-network input.

    The bounded input prior is created as follows:
      - finite raw value below 0%  -> 0%
      - finite raw value above 100% -> 100%
      - finite raw value in [0, 100] -> unchanged
      - non-finite raw value -> fallback_pct
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    df = pd.read_csv(input_path)

    required_cols = {
        "sample_id",
        "window_idx",
        raw_prior_col,
    }

    missing_cols = sorted(required_cols.difference(df.columns))
    if missing_cols:
        raise ValueError(
            f"Missing required columns in '{input_path}': {missing_cols}"
        )

    duplicate_count = int(
        df.duplicated(["sample_id", "window_idx"]).sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            "The raw operator file contains duplicate "
            f"(sample_id, window_idx) keys: {duplicate_count}"
        )

    raw_prior = pd.to_numeric(
        df[raw_prior_col],
        errors="coerce",
    ).to_numpy(dtype=float)

    finite_mask = np.isfinite(raw_prior)

    bounded_prior = np.full(
        raw_prior.shape,
        fill_value=float(fallback_pct),
        dtype=np.float64,
    )

    bounded_prior[finite_mask] = np.clip(
        raw_prior[finite_mask],
        0.0,
        100.0,
    )

    df[input_prior_col] = bounded_prior.astype(np.float32)
    df[FALLBACK_FLAG_COL] = (~finite_mask).astype(np.int8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    clipped_low_count = int(
        np.sum(finite_mask & (raw_prior < 0.0))
    )

    clipped_high_count = int(
        np.sum(finite_mask & (raw_prior > 100.0))
    )

    fallback_count = int(np.sum(~finite_mask))

    print(f"Saved model-input prior file: {output_path}")
    print(f"Total rows: {len(df)}")
    print(f"Finite raw priors: {int(np.sum(finite_mask))}")
    print(f"Fallback priors: {fallback_count}")
    print(f"Finite values clipped below 0%: {clipped_low_count}")
    print(f"Finite values clipped above 100%: {clipped_high_count}")
    print(f"New model-input column: {input_prior_col}")

    if raw_reason_col in df.columns:
        print("\nRaw operator reason counts:")
        print(
            df[raw_reason_col]
            .fillna("missing_reason")
            .value_counts()
            .to_string()
        )

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a bounded neural-network input alias from the raw "
            "two-ended positive-sequence operator output."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Raw two-ended operator CSV.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV containing the bounded model-input prior column.",
    )

    parser.add_argument(
        "--raw-prior-col",
        default=RAW_PRIOR_COL,
    )

    parser.add_argument(
        "--raw-reason-col",
        default=RAW_REASON_COL,
    )

    parser.add_argument(
        "--input-prior-col",
        default=INPUT_PRIOR_COL,
    )

    parser.add_argument(
        "--fallback-pct",
        type=float,
        default=50.0,
    )

    args = parser.parse_args()

    if not 0.0 <= args.fallback_pct <= 100.0:
        raise ValueError("--fallback-pct must be in the range [0, 100].")

    prepare_two_ended_prior_file(
        input_path=args.input,
        output_path=args.output,
        raw_prior_col=args.raw_prior_col,
        raw_reason_col=args.raw_reason_col,
        input_prior_col=args.input_prior_col,
        fallback_pct=args.fallback_pct,
    )


if __name__ == "__main__":
    main()
