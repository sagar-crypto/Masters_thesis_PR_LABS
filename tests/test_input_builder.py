from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from KOL.input_builder import (
    canonicalize,
    collect_candidates,
    normalize_sample_id,
    select_candidates,
)
from KOL.prepare_two_ended_prior_file import bound_prior_values


def test_bound_prior_values_clips_and_falls_back() -> None:
    values, fallback, counts = bound_prior_values([-1, 25, 101, np.nan, np.inf])
    assert values.tolist() == [0, 25, 100, 50, 50]
    assert fallback.tolist() == [False, False, False, True, True]
    assert counts == {"fallback": 2, "clipped_low": 1, "clipped_high": 1}


@pytest.mark.parametrize(
    ("value", "expected"),
    [(" 001 ", "1"), (1.0, "1"), ("event-a ", "event-a")],
)
def test_normalize_sample_id(value: object, expected: str) -> None:
    assert normalize_sample_id(value) == expected


def test_canonicalize_sorts_and_rejects_duplicate_normalized_keys() -> None:
    frame = pd.DataFrame({"sample_id": ["2", " 1 "], "window_idx": [9, 8]})
    assert canonicalize(frame, label="x").to_dict("records") == [
        {"sample_id": "1", "window_idx": 8},
        {"sample_id": "2", "window_idx": 9},
    ]
    duplicate = pd.DataFrame({"sample_id": ["1", 1.0], "window_idx": [8, 8]})
    with pytest.raises(ValueError, match="duplicate"):
        canonicalize(duplicate, label="x")


def test_candidate_deduplication_and_deterministic_tie_break() -> None:
    base = pd.DataFrame(
        {
            "sample_id": ["1", "2"],
            "window_idx": [8, 8],
            "case": ["a", "a"],
            "y_fault_location": [0.2, 0.8],
            "d_z_pct": [20.0, 80.0],
            "d_a_pct": [20.0, 80.0],
            "d_two_ended_posseq_plus_pct": [20.0, 80.0],
        }
    )
    # Identical vectors retain the canonical source role, then first column.
    candidates = collect_candidates(
        {"both": base, "default": base.assign(d_a_pct=[10.0, 90.0]), "opposite": base}
    )
    labels = [(candidate["role"], candidate["column"]) for candidate in candidates]
    assert labels == [("both", "d_a_pct"), ("default", "d_a_pct")]
    selected, mapping, _, _ = select_candidates(base, candidates, ["case"])
    assert selected.tolist() == [20.0, 80.0]
    assert mapping.loc[0, "selected_source_role"] == "both"
    assert mapping.loc[0, "selected_column"] == "d_a_pct"


def test_line_case_selection_and_nonfinite_target_rejection() -> None:
    reference = pd.DataFrame(
        {
            "sample_id": ["1", "2", "3", "4"],
            "window_idx": [8, 8, 8, 8],
            "line": ["x", "x", "y", "y"],
            "case": ["a", "a", "a", "a"],
            "y_true": [10.0, 20.0, 80.0, 90.0],
        }
    )
    candidates = [
        {"role": "both", "column": "d_low_pct", "values": np.array([10, 20, 10, 20])},
        {"role": "both", "column": "d_high_pct", "values": np.array([80, 90, 80, 90])},
    ]
    selected, mapping, _, _ = select_candidates(reference, candidates, ["line", "case"])
    assert selected.tolist() == [10, 20, 80, 90]
    assert mapping["selected_column"].tolist() == ["d_low_pct", "d_high_pct"]
    reference.loc[0, "y_true"] = np.nan
    with pytest.raises(ValueError, match="Target contains non-finite"):
        select_candidates(reference, candidates, ["line", "case"])
