from __future__ import annotations

import numpy as np


def get_line_side_mapping(fault_line: str, side_mode: str = "default") -> tuple[list[tuple[str, str]], str]:
    side_mode = str(side_mode).lower().strip()

    default_map = {
        "Line_1_2_a": [("Bus_1", "Line_01_02A")],
        "Line_1_2_b": [("Bus_1", "Line_01_02B")],
        "Line_2_3_a": [("Bus_2", "Line_02_03A")],
        "Line_2_3_b": [("Bus_2", "Line_02_03B")],
        "MainLn1-2A": [("Bus_1", "MainLn1-2A")],
        "MainLn1-2B": [("Bus_1", "MainLn1-2B")],
        "MainLn2-3A": [("Bus_2", "MainLn2-3A")],
        "MainLn2-3B": [("Bus_2", "MainLn2-3B")],
    }

    opposite_map = {
        "Line_1_2_a": [("Bus_2", "Line_01_02A")],
        "Line_1_2_b": [("Bus_2", "Line_01_02B")],
        "Line_2_3_a": [("Bus_3", "Line_02_03A")],
        "Line_2_3_b": [("Bus_3", "Line_02_03B")],
        "MainLn1-2A": [("Bus_2", "MainLn1-2A")],
        "MainLn1-2B": [("Bus_2", "MainLn1-2B")],
        "MainLn2-3A": [("Bus_3", "MainLn2-3A")],
        "MainLn2-3B": [("Bus_3", "MainLn2-3B")],
    }

    both_map = {k: default_map[k] + opposite_map[k] for k in default_map}

    if side_mode == "default":
        mapping = default_map
    elif side_mode == "opposite":
        mapping = opposite_map
    elif side_mode == "both":
        mapping = both_map
    else:
        raise ValueError(f"Unknown side_mode='{side_mode}'. Supported: default, opposite, both")

    if fault_line not in mapping:
        raise ValueError(f"Unknown fault_line: {fault_line}")

    return mapping[fault_line], side_mode


def extract_single_side_vi(
    x_raw: np.ndarray,
    feature_names: list[str],
    bus_token: str,
    line_token: str,
) -> np.ndarray:
    name_to_idx = {name: i for i, name in enumerate(feature_names)}

    candidates = [
        {
            "i1": f"{bus_token}_{line_token}_cur_L1_A",
            "i2": f"{bus_token}_{line_token}_cur_L2_A",
            "i3": f"{bus_token}_{line_token}_cur_L3_A",
            "v1": f"{bus_token}_{line_token}_vol_L1_V",
            "v2": f"{bus_token}_{line_token}_vol_L2_V",
            "v3": f"{bus_token}_{line_token}_vol_L3_V",
        },
        {
            "i1": f"{bus_token}_Other_{line_token}_Isec_L1_A",
            "i2": f"{bus_token}_Other_{line_token}_Isec_L2_A",
            "i3": f"{bus_token}_Other_{line_token}_Isec_L3_A",
            "v1": f"{bus_token}_Other_{line_token}_Usec_L1_V",
            "v2": f"{bus_token}_Other_{line_token}_Usec_L2_V",
            "v3": f"{bus_token}_Other_{line_token}_Usec_L3_V",
        },
    ]

    last_missing = None
    for cand in candidates:
        try:
            i1 = name_to_idx[cand["i1"]]
            i2 = name_to_idx[cand["i2"]]
            i3 = name_to_idx[cand["i3"]]
            v1 = name_to_idx[cand["v1"]]
            v2 = name_to_idx[cand["v2"]]
            v3 = name_to_idx[cand["v3"]]
            return np.stack(
                [
                    x_raw[:, v1],
                    x_raw[:, v2],
                    x_raw[:, v3],
                    x_raw[:, i1],
                    x_raw[:, i2],
                    x_raw[:, i3],
                ],
                axis=1,
            )
        except KeyError as e:
            last_missing = e

    raise KeyError(
        f"Could not find VI channels for bus_token={bus_token}, line_token={line_token}. "
        f"Last missing key: {last_missing}"
    )


def extract_line_vi_channels(
    x_raw: np.ndarray,
    feature_names: list[str],
    fault_line: str,
    side_mode: str = "default",
) -> tuple[np.ndarray, list[str]]:
    side_pairs, _ = get_line_side_mapping(fault_line=fault_line, side_mode=side_mode)

    chunks = []
    used_sides = []
    for bus_token, line_token in side_pairs:
        x_side = extract_single_side_vi(
            x_raw=x_raw,
            feature_names=feature_names,
            bus_token=bus_token,
            line_token=line_token,
        )
        chunks.append(x_side)
        used_sides.append(f"{bus_token}_{line_token}")

    x_vi = np.concatenate(chunks, axis=1)
    return x_vi, used_sides


def get_line_vi_channel_names(feature_names: list[str], fault_line: str) -> list[str]:
    line_map = {
        "Line_1_2_a": ("Bus_1", "Line_01_02A"),
        "Line_1_2_b": ("Bus_1", "Line_01_02B"),
        "Line_2_3_a": ("Bus_2", "Line_02_03A"),
        "Line_2_3_b": ("Bus_2", "Line_02_03B"),
        "MainLn1-2A": ("Bus_1", "MainLn1-2A"),
        "MainLn1-2B": ("Bus_1", "MainLn1-2B"),
        "MainLn2-3A": ("Bus_2", "MainLn2-3A"),
        "MainLn2-3B": ("Bus_2", "MainLn2-3B"),
    }

    if fault_line not in line_map:
        raise ValueError(f"Unknown fault_line: {fault_line}")

    bus_token, line_token = line_map[fault_line]

    candidates = [
        [
            f"{bus_token}_{line_token}_vol_L1_V",
            f"{bus_token}_{line_token}_vol_L2_V",
            f"{bus_token}_{line_token}_vol_L3_V",
            f"{bus_token}_{line_token}_cur_L1_A",
            f"{bus_token}_{line_token}_cur_L2_A",
            f"{bus_token}_{line_token}_cur_L3_A",
        ],
        [
            f"{bus_token}_Other_{line_token}_Usec_L1_V",
            f"{bus_token}_Other_{line_token}_Usec_L2_V",
            f"{bus_token}_Other_{line_token}_Usec_L3_V",
            f"{bus_token}_Other_{line_token}_Isec_L1_A",
            f"{bus_token}_Other_{line_token}_Isec_L2_A",
            f"{bus_token}_Other_{line_token}_Isec_L3_A",
        ],
    ]

    feature_set = set(feature_names)
    for cand in candidates:
        if all(name in feature_set for name in cand):
            return cand

    raise ValueError(f"No matching channel set found for fault_line={fault_line}")
