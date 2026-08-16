#!/bin/bash -l
#SBATCH --job-name=ch4_duble_ended_fresh_priors
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=./hpc/hpc_logs/%x/%x-%j-on-%N.out

set -euo pipefail

source /home/hpc/iwi5/iwi5305h/miniconda3/etc/profile.d/conda.sh
conda activate Masters_thesis_env_py312

THESIS_DIR="/home/hpc/iwi5/iwi5305h/Masters_thesis_PR_LABS"

RAW_BASE="$THESIS_DIR/outputs/chapter4/fresh_prior_exports/12098749_20260717_230756/raw_exports"

RAW_90="$RAW_BASE/hv_double_line_90kv/kol_operator_features_hv_double_line_90kv_all_lines_both_single_fault_start.csv"

RAW_110="$RAW_BASE/hv_double_line_110kv/kol_operator_features_hv_double_line_110kv_all_lines_both_all_fault_start.csv"

OUTPUT_BASE="$THESIS_DIR/outputs/chapter4/model_inputs/two_ended_posseq"

RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$OUTPUT_BASE/$RUN_TIMESTAMP"

mkdir -p "$OUTPUT_DIR"

export RAW_90
export RAW_110
export OUTPUT_DIR

python - <<'PY'
from pathlib import Path
import json
import os

import numpy as np
import pandas as pd


raw_90 = Path(os.environ["RAW_90"])
raw_110 = Path(os.environ["RAW_110"])
output_dir = Path(os.environ["OUTPUT_DIR"])

raw_column = "d_two_ended_posseq_plus_pct"
input_column = "d_two_ended_posseq_plus_input_pct"
fallback_column = "d_two_ended_posseq_plus_input_was_fallback"
clipped_column = "d_two_ended_posseq_plus_input_was_clipped"


def prepare(
    *,
    source: Path,
    output_name: str,
    expected_rows: int,
    expected_events: int,
    expected_windows: set[int] | None,
) -> tuple[Path, dict]:
    if not source.is_file():
        raise FileNotFoundError(source)

    dataframe = pd.read_csv(source)

    required = {
        "sample_id",
        "window_idx",
        raw_column,
    }

    missing = sorted(required - set(dataframe.columns))

    if missing:
        raise KeyError(
            f"{source} is missing columns: {missing}"
        )

    if len(dataframe) != expected_rows:
        raise ValueError(
            f"{source}: expected {expected_rows} rows, "
            f"found {len(dataframe)}"
        )

    event_count = dataframe["sample_id"].nunique(dropna=False)

    if event_count != expected_events:
        raise ValueError(
            f"{source}: expected {expected_events} events, "
            f"found {event_count}"
        )

    if dataframe.duplicated(["sample_id", "window_idx"]).any():
        raise ValueError(
            f"{source}: duplicate (sample_id, window_idx) keys"
        )

    if expected_windows is not None:
        observed_windows = set(
            pd.to_numeric(
                dataframe["window_idx"],
                errors="raise",
            ).astype(int).unique()
        )

        if observed_windows != expected_windows:
            raise ValueError(
                f"{source}: expected window_idx "
                f"{sorted(expected_windows)}, found "
                f"{sorted(observed_windows)}"
            )

        per_event = dataframe.groupby("sample_id").size()

        if not (
            per_event.min() == len(expected_windows)
            and per_event.max() == len(expected_windows)
        ):
            raise ValueError(
                f"{source}: expected exactly "
                f"{len(expected_windows)} rows per event; "
                f"observed min={per_event.min()}, "
                f"max={per_event.max()}"
            )

    raw_values = pd.to_numeric(
        dataframe[raw_column],
        errors="coerce",
    ).to_numpy(dtype=float)

    finite = np.isfinite(raw_values)

    bounded = np.full(
        len(dataframe),
        50.0,
        dtype=float,
    )

    bounded[finite] = np.clip(
        raw_values[finite],
        0.0,
        100.0,
    )

    fallback = ~finite

    clipped = (
        finite
        & (
            (raw_values < 0.0)
            | (raw_values > 100.0)
        )
    )

    dataframe[input_column] = bounded
    dataframe[fallback_column] = fallback.astype(np.int8)
    dataframe[clipped_column] = clipped.astype(np.int8)

    if not np.isfinite(
        dataframe[input_column].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            f"{source}: prepared prior contains non-finite values"
        )

    if not dataframe[input_column].between(
        0.0,
        100.0,
        inclusive="both",
    ).all():
        raise ValueError(
            f"{source}: prepared prior is outside [0,100]"
        )

    output_path = output_dir / output_name

    dataframe.to_csv(
        output_path,
        index=False,
    )

    summary = {
        "source": str(source),
        "output": str(output_path),
        "rows": int(len(dataframe)),
        "events": int(event_count),
        "finite_raw": int(finite.sum()),
        "invalid_raw": int((~finite).sum()),
        "clipped_low": int(
            np.sum(finite & (raw_values < 0.0))
        ),
        "clipped_high": int(
            np.sum(finite & (raw_values > 100.0))
        ),
        "fallback_count": int(fallback.sum()),
        "prepared_prior_min_pp": float(
            dataframe[input_column].min()
        ),
        "prepared_prior_max_pp": float(
            dataframe[input_column].max()
        ),
        "prior_column": input_column,
    }

    return output_path, summary


output_90, summary_90 = prepare(
    source=raw_90,
    output_name=(
        "kol_operator_features_hv_double_line_90kv_"
        "two_ended_posseq_single_fault_start_model_input.csv"
    ),
    expected_rows=9022,
    expected_events=9022,
    expected_windows=None,
)

output_110, summary_110 = prepare(
    source=raw_110,
    output_name=(
        "kol_operator_features_hv_double_line_110kv_"
        "two_ended_posseq_all_fault_start_model_input.csv"
    ),
    expected_rows=3648,
    expected_events=912,
    expected_windows={8, 9, 10, 11},
)

summary = {
    "90kv": summary_90,
    "110kv": summary_110,
}

summary_path = output_dir / "two_ended_input_summary.json"

summary_path.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)

print("=" * 80)
print("TWO-ENDED MODEL INPUTS CREATED")
print("=" * 80)

for topology, values in summary.items():
    print()
    print(topology)
    for key, value in values.items():
        print(f"  {key}: {value}")

print()
print("Summary:")
print(summary_path)
PY

P90_2E_PRIOR="$OUTPUT_DIR/kol_operator_features_hv_double_line_90kv_two_ended_posseq_single_fault_start_model_input.csv"

P110_2E_PRIOR="$OUTPUT_DIR/kol_operator_features_hv_double_line_110kv_two_ended_posseq_all_fault_start_model_input.csv"

ENV_FILE="$OUTPUT_DIR/two_ended_inputs.env"

cat > "$ENV_FILE" <<EOF
TWO_ENDED_INPUT_DIR='$OUTPUT_DIR'

P90_2E_PRIOR='$P90_2E_PRIOR'
P90_2E_PRIOR_COL='d_two_ended_posseq_plus_input_pct'
P90_2E_OPERATOR_FEATURE_COLS='[]'

P110_2E_PRIOR='$P110_2E_PRIOR'
P110_2E_PRIOR_COL='d_two_ended_posseq_plus_input_pct'
P110_2E_OPERATOR_FEATURE_COLS='[]'
EOF

cp \
  "$ENV_FILE" \
  "$OUTPUT_BASE/LATEST_TWO_ENDED_INPUTS.env"

printf '%s\n' \
  "$OUTPUT_DIR" \
  > "$OUTPUT_BASE/LATEST_TWO_ENDED_INPUT_DIR.txt"

echo
echo "============================================================"
echo "PREPARATION COMPLETED"
echo "============================================================"
echo "Output directory:"
echo "$OUTPUT_DIR"
echo
echo "Latest environment:"
echo "$OUTPUT_BASE/LATEST_TWO_ENDED_INPUTS.env"
echo "============================================================"
BASH
