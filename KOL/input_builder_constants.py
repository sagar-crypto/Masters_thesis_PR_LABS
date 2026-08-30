"""Static schema and cohort configuration for Chapter 4 input building."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

KEYS = ["sample_id", "window_idx"]
TARGET_ALIASES = ["y_fault_location", "y_true", "fault_location"]
CASE_ALIASES = ["case", "fault_case", "sc_type", "y_fault_case", "fault_type"]
LINE_ALIASES = ["y_fault_line", "fault_line", "line", "line_name"]
FEATURES_90 = {
    "ratio_V0_V1": ["ratio_V0_V1", "ratio_V0_V1_local"],
    "ratio_V2_V1": ["ratio_V2_V1", "ratio_V2_V1_local"],
    "ratio_I0_I1": ["ratio_I0_I1", "ratio_I0_I1_local"],
    "ratio_I2_I1": ["ratio_I2_I1", "ratio_I2_I1_local"],
    "abs_Z0_app": ["abs_Z0_app", "abs_Z0_app_local"],
    "abs_Z2_app": ["abs_Z2_app", "abs_Z2_app_local"],
}
FEATURES_110 = [
    "d_both_diff_real_pct",
    "ratio_V0_V1_local",
    "ratio_V2_V1_local",
    "ratio_I0_I1_local",
    "ratio_I2_I1_local",
    "abs_Z0_app_local",
    "abs_Z2_app_local",
    "ratio_V0_V1_remote",
    "ratio_V2_V1_remote",
    "ratio_I0_I1_remote",
    "ratio_I2_I1_remote",
    "abs_Z0_app_remote",
    "abs_Z2_app_remote",
]
EXCLUDED = {
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
    "fusion",
}
ROLE_ORDER = {"both": 0, "default": 1, "opposite": 2}


class CohortSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    rows: int
    events: int
    window_indices: set[int]
    event_windows: set[int]


SPECS = {
    "90": CohortSpec(
        rows=40564,
        events=9022,
        window_indices={8, 9, 10, 11, 12},
        event_windows={4, 5},
    ),
    "110": CohortSpec(
        rows=3648,
        events=912,
        window_indices={8, 9, 10, 11},
        event_windows={4},
    ),
}
