from __future__ import annotations

HV110KV_LINE_PARAMS = {
    "MainLn1-2A": {
        "r1": 0.6678496599197388,
        "x1": 5.200491428375244,
        "r0": 2.362898349761963,
        "x0": 19.572376251220703,
        "length": 20.0,
    },
    "MainLn1-2B": {
        "r1": 0.8348121047019958,
        "x1": 6.500614643096924,
        "r0": 2.953623056411743,
        "x0": 24.465471267700195,
        "length": 25.0,
    },
    "MainLn2-3A": {
        "r1": 1.001774549484253,
        "x1": 7.800737380981445,
        "r0": 3.5443475246429443,
        "x0": 29.358566284179688,
        "length": 30.0,
    },
    "MainLn2-3B": {
        "r1": 1.3356993198394775,
        "x1": 10.400982856750488,
        "r0": 4.725796699523926,
        "x0": 39.144752502441406,
        "length": 40.0,
    },
}

CASE_TO_IDX = {
    "3ph": 0,
    "slg_a": 1,
    "slg_b": 2,
    "slg_c": 3,
    "ll_ab": 4,
    "ll_bc": 5,
    "ll_ca": 6,
    "llg_ab": 7,
    "llg_bc": 8,
    "llg_ca": 9,
}

ALL_CASE_TO_IDX = CASE_TO_IDX.copy()

GROUND_CASE_IDS = {
    CASE_TO_IDX["slg_a"],
    CASE_TO_IDX["slg_b"],
    CASE_TO_IDX["slg_c"],
    CASE_TO_IDX["llg_ab"],
    CASE_TO_IDX["llg_bc"],
    CASE_TO_IDX["llg_ca"],
}

GROUND_CASES = {"slg_a", "slg_b", "slg_c", "llg_ab", "llg_bc", "llg_ca"}
LL_CASES = {"ll_ab", "ll_bc", "ll_ca"}
THREEPH_CASES = {"3ph"}
