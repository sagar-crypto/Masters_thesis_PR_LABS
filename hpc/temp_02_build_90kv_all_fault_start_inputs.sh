#!/bin/bash -l
#SBATCH --job-name=ch4_fresh_priors
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=./hpc/hpc_logs/tmp90_afs_physics-%j-on-%N.out
#SBATCH --error=./hpc/hpc_logs/tmp90_afs_physics-%j-on-%N.err

set -euo pipefail


source \
  /home/hpc/iwi5/iwi5305h/miniconda3/etc/profile.d/conda.sh

conda activate Masters_thesis_env_py312


export THESIS_DIR="${THESIS_DIR:-/home/hpc/iwi5/iwi5305h/Masters_thesis_PR_LABS}"

export TEMP_BASE=\
"$THESIS_DIR/outputs/chapter4/temp_90kv_afs_check"

export RAW_POINTER=\
"$TEMP_BASE/LATEST_RAW_DIR.txt"


if [ ! -s "$RAW_POINTER" ]; then
    echo "ERROR: Missing raw-export pointer:"
    echo "$RAW_POINTER"

    echo
    echo "Run this first:"
    echo "sbatch hpc/temp_01_export_90kv_all_fault_start_physics.sh"

    exit 1
fi


export RAW_DIR="$(
    head -n 1 "$RAW_POINTER"
)"


if [ ! -d "$RAW_DIR" ]; then
    echo "ERROR: Raw export directory does not exist:"
    echo "$RAW_DIR"
    exit 1
fi


RUN_TAG="$(
    date +%Y%m%d_%H%M%S
)"


export INPUT_DIR=\
"$TEMP_BASE/model_inputs/$RUN_TAG"

export INPUT_ENV=\
"$INPUT_DIR/temp_90kv_afs_inputs.env"

export LATEST_ENV=\
"$TEMP_BASE/LATEST_INPUTS.env"


mkdir -p "$INPUT_DIR"


python - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import shlex

from pathlib import Path

import numpy as np
import pandas as pd


raw_dir = Path(
    os.environ["RAW_DIR"]
)

input_dir = Path(
    os.environ["INPUT_DIR"]
)

input_env = Path(
    os.environ["INPUT_ENV"]
)

latest_env = Path(
    os.environ["LATEST_ENV"]
)


TARGET_ALIASES = [
    "y_fault_location",
    "y_true",
    "fault_location",
]

CASE_ALIASES = [
    "case",
    "fault_case",
    "sc_type",
    "fault_type",
]

LINE_ALIASES = [
    "y_fault_line",
    "fault_line",
    "line",
]


FEATURE_ALIASES = {
    "ratio_V0_V1": [
        "ratio_V0_V1",
        "ratio_V0_V1_local",
    ],
    "ratio_V2_V1": [
        "ratio_V2_V1",
        "ratio_V2_V1_local",
    ],
    "ratio_I0_I1": [
        "ratio_I0_I1",
        "ratio_I0_I1_local",
    ],
    "ratio_I2_I1": [
        "ratio_I2_I1",
        "ratio_I2_I1_local",
    ],
    "abs_Z0_app": [
        "abs_Z0_app",
        "abs_Z0_app_local",
    ],
    "abs_Z2_app": [
        "abs_Z2_app",
        "abs_Z2_app_local",
    ],
}


ONE_ENDED_EXCLUDED_TOKENS = {
    "error",
    "residual",
    "target",
    "true",
    "reason",
    "valid",
    "fallback",
    "flag",
    "diff",
    "ratio",
    "weight",
    "confidence",
    "score",
    "improvement",
    "two_ended",
    "twoended",
    "posseq_plus",
    "both_mean",
    "both_weighted",
    "both_edge",
    "both_min",
    "both_max",
    "disagreement",
    "fusion",
}


def canonical_sample_id(
    value,
):
    if pd.isna(value):
        return "<missing>"

    text = str(value).strip()

    try:
        number = float(text)

        if (
            np.isfinite(number)
            and number.is_integer()
        ):
            return str(int(number))

    except Exception:
        pass

    return text


def canonical(
    frame: pd.DataFrame,
    path: Path,
) -> pd.DataFrame:
    required = {
        "sample_id",
        "window_idx",
    }

    missing = sorted(
        required
        - set(frame.columns)
    )

    if missing:
        raise KeyError(
            f"{path}: missing key "
            f"columns {missing}"
        )

    output = frame.copy()

    output["sample_id"] = (
        output["sample_id"]
        .map(
            canonical_sample_id
        )
    )

    output["window_idx"] = (
        pd.to_numeric(
            output["window_idx"],
            errors="raise",
        )
        .astype(int)
    )

    output = (
        output.sort_values(
            [
                "sample_id",
                "window_idx",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    if output.duplicated(
        [
            "sample_id",
            "window_idx",
        ]
    ).any():
        raise ValueError(
            f"{path}: duplicate "
            "sample/window keys"
        )

    return output


def first_existing(
    frame: pd.DataFrame,
    aliases: list[str],
    label: str,
) -> str:
    for column in aliases:
        if column in frame.columns:
            return column

    raise KeyError(
        f"Could not resolve {label}; "
        f"tried {aliases}"
    )


def target_to_pp(
    values: pd.Series,
) -> np.ndarray:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).to_numpy(dtype=float)

    finite = numeric[
        np.isfinite(numeric)
    ]

    if len(finite) == 0:
        raise ValueError(
            "Target column contains "
            "no finite values"
        )

    if np.max(
        np.abs(finite)
    ) <= 1.5:
        numeric = numeric * 100.0

    if not np.isfinite(
        numeric
    ).all():
        raise ValueError(
            "Target column contains "
            "non-finite values"
        )

    return numeric


def prepare_prior(
    raw: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    raw = np.asarray(
        raw,
        dtype=float,
    )

    finite = np.isfinite(raw)

    prepared = np.full(
        len(raw),
        50.0,
        dtype=float,
    )

    prepared[finite] = np.clip(
        raw[finite],
        0.0,
        100.0,
    )

    return (
        prepared,
        ~finite,
    )


def vector_hash(
    values: np.ndarray,
) -> str:
    normalized = np.nan_to_num(
        np.asarray(
            values,
            dtype=np.float64,
        ),
        nan=1e30,
        posinf=1e31,
        neginf=-1e31,
    )

    return hashlib.sha256(
        normalized.tobytes()
    ).hexdigest()


def is_one_ended_candidate(
    column: str,
    source_path: Path,
) -> bool:
    name = column.lower()

    if (
        not name.startswith("d_")
        or "pct" not in name
    ):
        return False

    if any(
        token in name
        for token
        in ONE_ENDED_EXCLUDED_TOKENS
    ):
        return False

    # In the "both" export, d_phys_real_pct is not treated as
    # an independent one-ended candidate.
    if (
        name == "d_phys_real_pct"
        and "_both_"
        in source_path.name.lower()
    ):
        return False

    return True


paths = sorted(
    raw_dir.glob("*.csv")
)


if len(paths) != 3:
    raise RuntimeError(
        "Expected three raw exports "
        "(default, opposite, and both), "
        f"found {len(paths)}"
    )


frames: list[
    tuple[
        Path,
        pd.DataFrame,
    ]
] = []


reference_keys = None


for path in paths:
    frame = canonical(
        pd.read_csv(path),
        path,
    )

    keys = frame[
        [
            "sample_id",
            "window_idx",
        ]
    ]

    if reference_keys is None:
        reference_keys = keys

    elif not keys.equals(
        reference_keys
    ):
        raise RuntimeError(
            f"Key mismatch for "
            f"{path.name}"
        )

    frames.append(
        (
            path,
            frame,
        )
    )


reference_candidates = []


for path, frame in frames:
    if (
        any(
            column in frame.columns
            for column
            in TARGET_ALIASES
        )
        and any(
            column in frame.columns
            for column
            in CASE_ALIASES
        )
        and any(
            column in frame.columns
            for column
            in LINE_ALIASES
        )
    ):
        reference_candidates.append(
            (
                len(frame.columns),
                path,
                frame,
            )
        )


if not reference_candidates:
    raise RuntimeError(
        "No metadata-rich export "
        "was found"
    )


(
    _,
    reference_path,
    reference,
) = max(
    reference_candidates,
    key=lambda row: row[0],
)


target_column = first_existing(
    reference,
    TARGET_ALIASES,
    "target",
)

case_column = first_existing(
    reference,
    CASE_ALIASES,
    "fault case",
)

line_column = first_existing(
    reference,
    LINE_ALIASES,
    "faulted line",
)


target_pp = target_to_pp(
    reference[target_column]
)

case_values = (
    reference[case_column]
    .astype(str)
    .str.strip()
    .str.lower()
)


event_sizes = (
    reference.groupby(
        "sample_id"
    )
    .size()
)


if event_sizes.size != 9022:
    raise RuntimeError(
        "Expected 9022 physical events, "
        f"found {event_sizes.size}"
    )


if int(event_sizes.min()) < 1:
    raise RuntimeError(
        "At least one physical event has "
        "no retained fault-start window"
    )

print(
    "Windows per event: "
    f"min={int(event_sizes.min())}, "
    f"max={int(event_sizes.max())}, "
    f"mean={float(event_sizes.mean()):.3f}"
)


# ------------------------------------------------------------------
# Resolve the six scalar operator features for the 1E model input.
# ------------------------------------------------------------------

features: dict[
    str,
    np.ndarray,
] = {}


for (
    output_name,
    aliases,
) in FEATURE_ALIASES.items():
    resolved = None

    for _, frame in frames:
        for alias in aliases:
            if alias not in frame.columns:
                continue

            values = pd.to_numeric(
                frame[alias],
                errors="coerce",
            ).to_numpy(dtype=float)

            if np.isfinite(
                values
            ).all():
                resolved = values
                break

        if resolved is not None:
            break

    if resolved is None:
        raise RuntimeError(
            "Could not resolve finite "
            f"feature {output_name}"
        )

    features[
        output_name
    ] = resolved


# ------------------------------------------------------------------
# Build the eligible one-ended candidate bank.
# ------------------------------------------------------------------

one_ended_candidates: dict[
    str,
    dict,
] = {}

seen_vectors: set[str] = set()


for path, frame in frames:
    for column in frame.columns:
        if not is_one_ended_candidate(
            column,
            path,
        ):
            continue

        raw_values = pd.to_numeric(
            frame[column],
            errors="coerce",
        ).to_numpy(dtype=float)

        fingerprint = vector_hash(
            raw_values
        )

        if fingerprint in seen_vectors:
            continue

        seen_vectors.add(
            fingerprint
        )

        label = (
            f"{path.name}::{column}"
        )

        one_ended_candidates[
            label
        ] = {
            "label": label,
            "source_file": (
                path.name
            ),
            "column": column,
            "raw_values": (
                raw_values
            ),
        }


if not one_ended_candidates:
    raise RuntimeError(
        "No eligible one-ended "
        "distance candidates were found"
    )


selected_1e = np.full(
    len(reference),
    np.nan,
    dtype=float,
)

selected_1e_raw = np.full(
    len(reference),
    np.nan,
    dtype=float,
)

selected_1e_label = np.empty(
    len(reference),
    dtype=object,
)

selected_1e_fallback = np.zeros(
    len(reference),
    dtype=np.int8,
)

mapping_rows = []


# ------------------------------------------------------------------
# Select the lowest-MAE candidate separately for each known fault case.
#
# IMPORTANT:
# This is a full-cohort, target-informed temporary sensitivity check.
# It must not be presented as a deployment-ready prior.
# ------------------------------------------------------------------

for (
    case_name,
    indices,
) in reference.groupby(
    case_values,
    sort=True,
).indices.items():
    indices = np.asarray(
        indices,
        dtype=int,
    )

    rankings = []

    for (
        label,
        candidate,
    ) in one_ended_candidates.items():
        raw_group = (
            candidate[
                "raw_values"
            ][indices]
        )

        (
            prepared_group,
            fallback_group,
        ) = prepare_prior(
            raw_group
        )

        errors = np.abs(
            prepared_group
            - target_pp[indices]
        )

        rankings.append(
            {
                "candidate": label,
                "source_file": (
                    candidate[
                        "source_file"
                    ]
                ),
                "column": (
                    candidate[
                        "column"
                    ]
                ),
                "mae_pp": float(
                    np.mean(errors)
                ),
                "rmse_pp": float(
                    np.sqrt(
                        np.mean(
                            errors ** 2
                        )
                    )
                ),
                "fallback_count": int(
                    fallback_group.sum()
                ),
            }
        )

    ranking = (
        pd.DataFrame(
            rankings
        )
        .sort_values(
            [
                "mae_pp",
                "fallback_count",
                "candidate",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    best = ranking.iloc[0]

    best_candidate = (
        one_ended_candidates[
            str(
                best["candidate"]
            )
        ]
    )

    raw_group = (
        best_candidate[
            "raw_values"
        ][indices]
    )

    (
        prepared_group,
        fallback_group,
    ) = prepare_prior(
        raw_group
    )

    selected_1e[
        indices
    ] = prepared_group

    selected_1e_raw[
        indices
    ] = raw_group

    selected_1e_label[
        indices
    ] = str(
        best["candidate"]
    )

    selected_1e_fallback[
        indices
    ] = fallback_group.astype(
        np.int8
    )

    mapping_rows.append(
        {
            "case": str(
                case_name
            ),
            "rows": int(
                len(indices)
            ),
            "selected_candidate": str(
                best[
                    "candidate"
                ]
            ),
            "selected_source_file": str(
                best[
                    "source_file"
                ]
            ),
            "selected_column": str(
                best[
                    "column"
                ]
            ),
            "selected_mae_pp": float(
                best[
                    "mae_pp"
                ]
            ),
            "selected_rmse_pp": float(
                best[
                    "rmse_pp"
                ]
            ),
            "fallback_count": int(
                best[
                    "fallback_count"
                ]
            ),
        }
    )


if not np.isfinite(
    selected_1e
).all():
    raise RuntimeError(
        "The selected 1E prepared prior "
        "contains non-finite values"
    )


# ------------------------------------------------------------------
# Prepare the synchronized two-ended positive-sequence prior.
#
# There is one true 2E formula here, so there is no target-based
# formula selection for the two-ended input.
# ------------------------------------------------------------------

two_ended_sources = []


for path, frame in frames:
    if (
        "d_two_ended_posseq_plus_pct"
        in frame.columns
    ):
        two_ended_sources.append(
            (
                path,
                frame,
            )
        )


if len(
    two_ended_sources
) != 1:
    raise RuntimeError(
        "Expected exactly one "
        "synchronized two-ended source "
        "containing "
        "d_two_ended_posseq_plus_pct; "
        f"found {len(two_ended_sources)}"
    )


(
    two_ended_path,
    two_ended_frame,
) = two_ended_sources[0]


raw_2e = pd.to_numeric(
    two_ended_frame[
        "d_two_ended_posseq_plus_pct"
    ],
    errors="coerce",
).to_numpy(dtype=float)


(
    prepared_2e,
    fallback_2e,
) = prepare_prior(
    raw_2e
)


if not np.isfinite(
    prepared_2e
).all():
    raise RuntimeError(
        "The prepared 2E prior "
        "contains non-finite values"
    )


# ------------------------------------------------------------------
# Write target-free training CSVs.
# ------------------------------------------------------------------

key_frame = reference[
    [
        "sample_id",
        "window_idx",
    ]
].copy()


one_ended_column = (
    "d_90kv_afs_case_bestmae_input_pct"
)

two_ended_column = (
    "d_90kv_afs_two_ended_posseq_input_pct"
)


one_ended_output = (
    key_frame.copy()
)

one_ended_output[
    one_ended_column
] = selected_1e


for (
    feature_name,
    values,
) in features.items():
    one_ended_output[
        feature_name
    ] = values


two_ended_output = (
    key_frame.copy()
)

two_ended_output[
    two_ended_column
] = prepared_2e


one_ended_path = (
    input_dir
    / (
        "kol_operator_features_"
        "hv_double_line_90kv_"
        "case_bestmae_"
        "all_fault_start_"
        "model_input.csv"
    )
)

two_ended_output_path = (
    input_dir
    / (
        "kol_operator_features_"
        "hv_double_line_90kv_"
        "two_ended_posseq_"
        "all_fault_start_"
        "model_input.csv"
    )
)


one_ended_output.to_csv(
    one_ended_path,
    index=False,
)

two_ended_output.to_csv(
    two_ended_output_path,
    index=False,
)


mapping_frame = pd.DataFrame(
    mapping_rows
)

mapping_frame.to_csv(
    input_dir
    / "one_ended_case_mapping_audit.csv",
    index=False,
)


row_audit = reference[
    [
        "sample_id",
        "window_idx",
        line_column,
        case_column,
        target_column,
    ]
].copy()


row_audit[
    "selected_1e_candidate"
] = selected_1e_label

row_audit[
    "selected_1e_raw_pct"
] = selected_1e_raw

row_audit[
    "selected_1e_input_pct"
] = selected_1e

row_audit[
    "selected_1e_fallback"
] = selected_1e_fallback

row_audit[
    "raw_2e_pct"
] = raw_2e

row_audit[
    "input_2e_pct"
] = prepared_2e

row_audit[
    "input_2e_fallback"
] = fallback_2e.astype(
    np.int8
)


row_audit.to_csv(
    input_dir
    / "temporary_prior_row_audit.csv",
    index=False,
)


minimum_windows_per_event = int(
    event_sizes.min()
)

maximum_windows_per_event = int(
    event_sizes.max()
)

mean_windows_per_event = float(
    event_sizes.mean()
)


summary = {
    "status": "PASS",
    "scope": (
        "temporary_90kv_"
        "all_fault_start_check"
    ),
    "selection_disclosure": (
        "The 1E prior selects the "
        "lowest-MAE eligible one-ended "
        "operator per fault case using "
        "the complete temporary cohort "
        "and true target."
    ),
    "two_ended_disclosure": (
        "The 2E input uses the single "
        "synchronized positive-sequence "
        "formula; no target-based 2E "
        "formula selection is performed."
    ),
    "reference_source": str(
        reference_path
    ),
    "rows": int(
        len(reference)
    ),
    "events": int(
        reference[
            "sample_id"
        ].nunique()
    ),
    "minimum_windows_per_event": (
        minimum_windows_per_event
    ),
    "maximum_windows_per_event": (
        maximum_windows_per_event
    ),
    "mean_windows_per_event": (
        mean_windows_per_event
    ),
    "window_indices": sorted(
        reference[
            "window_idx"
        ]
        .unique()
        .tolist()
    ),
    "one_ended_candidate_count": int(
        len(
            one_ended_candidates
        )
    ),
    "one_ended_input": str(
        one_ended_path
    ),
    "one_ended_column": (
        one_ended_column
    ),
    "one_ended_features": list(
        features
    ),
    "one_ended_prior_mae_pp": float(
        np.mean(
            np.abs(
                selected_1e
                - target_pp
            )
        )
    ),
    "two_ended_source": str(
        two_ended_path
    ),
    "two_ended_input": str(
        two_ended_output_path
    ),
    "two_ended_column": (
        two_ended_column
    ),
    "two_ended_fallback_count": int(
        fallback_2e.sum()
    ),
    "two_ended_prior_mae_pp": float(
        np.mean(
            np.abs(
                prepared_2e
                - target_pp
            )
        )
    ),
}


(
    input_dir
    / "temporary_input_summary.json"
).write_text(
    json.dumps(
        summary,
        indent=2,
    ),
    encoding="utf-8",
)


feature_list = (
    "["
    + ",".join(
        features
    )
    + "]"
)


environment_lines = [
    (
        "TEMP_90_AFS_INPUT_DIR="
        + shlex.quote(
            str(input_dir)
        )
    ),
    (
        "P90_1E_AFS_PRIOR="
        + shlex.quote(
            str(
                one_ended_path
            )
        )
    ),
    (
        "P90_1E_AFS_PRIOR_COL="
        + shlex.quote(
            one_ended_column
        )
    ),
    (
        "P90_1E_AFS_FEATURE_COLS="
        + shlex.quote(
            feature_list
        )
    ),
    (
        "P90_2E_AFS_PRIOR="
        + shlex.quote(
            str(
                two_ended_output_path
            )
        )
    ),
    (
        "P90_2E_AFS_PRIOR_COL="
        + shlex.quote(
            two_ended_column
        )
    ),
    "P90_2E_AFS_FEATURE_COLS='[]'",
]


environment_text = (
    "\n".join(
        environment_lines
    )
    + "\n"
)


input_env.write_text(
    environment_text,
    encoding="utf-8",
)


latest_tmp = (
    latest_env.with_suffix(
        ".tmp"
    )
)

latest_tmp.write_text(
    environment_text,
    encoding="utf-8",
)

latest_tmp.replace(
    latest_env
)


print()
print("=" * 90)
print(
    "TEMPORARY 90 kV "
    "ALL-FAULT-START INPUTS COMPLETE"
)
print("=" * 90)

print(
    json.dumps(
        summary,
        indent=2,
    )
)

print()
print("Environment file:")
print(latest_env)

print()
print(environment_text)
PY


echo
echo "Generated files:"

find "$INPUT_DIR" \
    -maxdepth 1 \
    -type f \
    -printf '%f\t%s bytes\n' \
    | sort


echo
echo "Latest environment:"

cat "$LATEST_ENV"
