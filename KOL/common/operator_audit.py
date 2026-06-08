from __future__ import annotations

from typing import cast

import pandas as pd

from KOL.common.cases import (
    derive_fault_case_from_processed_labels,
    formula_name_for_case,
)
from KOL.common.channel_mapping import get_line_vi_channel_names


def audit_case_and_formula_mapping(
    df: pd.DataFrame,
    feature_names: list[str],
    max_print: int | None = 50,
) -> pd.DataFrame:
    rows = []

    for i in range(len(df)):
        row = cast(pd.Series, df.iloc[i])

        case = derive_fault_case_from_processed_labels(row)
        formula = formula_name_for_case(case)

        try:
            channel_names = get_line_vi_channel_names(
                feature_names=feature_names,
                fault_line=str(row["y_fault_line"]),
            )
            channel_ok = True
            channel_error = ""
        except Exception as e:
            channel_names = []
            channel_ok = False
            channel_error = str(e)

        a = int(row["y_phase_A"])
        b = int(row["y_phase_B"])
        c = int(row["y_phase_C"])
        grounded = int(row["y_is_grounded"])

        active = []
        if a == 1:
            active.append("a")
        if b == 1:
            active.append("b")
        if c == 1:
            active.append("c")

        if len(active) == 3:
            expected_case = "3ph"
        elif len(active) == 1 and grounded == 1:
            expected_case = f"slg_{active[0]}"
        elif len(active) == 2 and grounded == 0:
            pair = "".join(active)
            if pair == "ac":
                pair = "ca"
            expected_case = f"ll_{pair}"
        elif len(active) == 2 and grounded == 1:
            pair = "".join(active)
            if pair == "ac":
                pair = "ca"
            expected_case = f"llg_{pair}"
        else:
            expected_case = "invalid"

        case_ok = case == expected_case

        rows.append(
            {
                "row_idx": i,
                "sample_id": row["sample_id"],
                "status": row["status"],
                "fault_line": row["y_fault_line"],
                "fault_location_pct": float(row["y_fault_location"]),
                "dt_start": float(row["dt_start"]),
                "y_phase_A": a,
                "y_phase_B": b,
                "y_phase_C": c,
                "y_is_grounded": grounded,
                "derived_case": case,
                "expected_case": expected_case,
                "case_ok": case_ok,
                "formula": formula,
                "channel_ok": channel_ok,
                "channel_error": channel_error,
                "channels": " | ".join(channel_names),
            }
        )

    audit_df = pd.DataFrame(rows)

    print("\n===== CASE / FORMULA AUDIT =====")
    print(f"Total rows audited: {len(audit_df)}")
    print(f"Rows with case mismatch: {(~audit_df['case_ok']).sum()}")
    print(f"Rows with channel mapping error: {(~audit_df['channel_ok']).sum()}")

    if max_print is not None:
        print("\nFirst audited rows:")
        print(audit_df.head(max_print).to_string(index=False))

    return audit_df
