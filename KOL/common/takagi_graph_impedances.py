from __future__ import annotations

import pickle
from typing import Any

import networkx as nx


def line_impedances(G: nx.MultiGraph) -> dict[str, tuple[complex, complex]]:
    """
    Return {line_name: (Z1_line, Z0_line)} as complex ohms.

    In this graph/pipeline, rline/xline/rline0/xline0 already match the
    operator CSV r1/x1/r0/x0 values, so do NOT multiply by length.
    """
    out: dict[str, tuple[complex, complex]] = {}

    for _u, _v, key, attr in G.edges(keys=True, data=True):
        pdict = attr.get(f"{key}_param_dict", {})

        if "rline" not in pdict:
            continue

        Z1_line = complex(float(pdict["rline"]), float(pdict["xline"]))
        Z0_line = complex(float(pdict["rline0"]), float(pdict["xline0"]))

        out[str(key)] = (Z1_line, Z0_line)

    return out


def extgrid_source_impedance(param: dict[str, Any], Un_kV: float) -> tuple[complex, complex]:
    """
    Compute external-grid positive- and zero-sequence source impedance
    at the external-grid voltage level.

    PowerFactory stores external-grid source strength using:
        snss   -> short-circuit power Sk'' in MVA
        rntxn  -> R1/X1 ratio
        x0tx1  -> X0/X1 ratio
        r0tx0  -> R0/X0 ratio
    """
    Sk = float(param["snss"])
    rx = float(param["rntxn"])
    x0x1 = float(param["x0tx1"])
    r0x0 = float(param["r0tx0"])

    Z1_mag = Un_kV**2 / Sk

    X1 = Z1_mag / (1.0 + rx**2) ** 0.5
    R1 = rx * X1
    Z1_src = complex(R1, X1)

    X0 = x0x1 * X1
    R0 = r0x0 * X0
    Z0_src = complex(R0, X0)

    return Z1_src, Z0_src


def transformer_impedance(typ: dict[str, Any], Un_l_kV: float) -> complex:
    """
    Approximate positive-sequence transformer impedance referred to LV side.

    Uses:
        uktr  -> short-circuit voltage in %
        strn  -> rated power in MVA
        pcutr -> copper losses in kW
    """
    uk = float(typ["uktr"])
    Sn = float(typ["strn"])
    pcu = float(typ["pcutr"])

    Zt_abs = (uk / 100.0) * Un_l_kV**2 / Sn
    Rt = (pcu / 1000.0) * Un_l_kV**2 / Sn**2

    Xt_sq = Zt_abs**2 - Rt**2
    Xt = Xt_sq**0.5 if Xt_sq > 0.0 else 0.0

    return complex(Rt, Xt)

def transformer_impedance_from_edge_param(
    pdict: dict,
    Un_l_kV: float = 110.0,
) -> tuple[complex, complex]:
    """
    Compute transformer positive- and zero-sequence impedance from the edge
    parameter dictionary used in this graph.

    Graph fields:
        uktr  -> positive-sequence short-circuit voltage %
        uktrr -> positive-sequence resistive voltage drop %
        uk0tr -> zero-sequence short-circuit voltage %
        ur0tr -> zero-sequence resistive voltage drop %
        strn  -> rated power in MVA
    """
    Sn = float(pdict["strn"])

    z1_abs = (float(pdict["uktr"]) / 100.0) * Un_l_kV**2 / Sn
    r1 = (float(pdict["uktrr"]) / 100.0) * Un_l_kV**2 / Sn
    x1 = max(z1_abs**2 - r1**2, 0.0) ** 0.5

    z0_abs = (float(pdict["uk0tr"]) / 100.0) * Un_l_kV**2 / Sn
    r0 = (float(pdict["ur0tr"]) / 100.0) * Un_l_kV**2 / Sn
    x0 = max(z0_abs**2 - r0**2, 0.0) ** 0.5

    return complex(r1, x1), complex(r0, x0)

def find_extgrid_param_dict_on_node(G: nx.MultiGraph, extgrid_node: str) -> dict | None:
    """
    Try to find external-grid source parameters on a node.

    Expected keys from the extraction script:
        snss, rntxn, x0tx1, r0tx0
    """
    required = {"snss", "rntxn", "x0tx1", "r0tx0"}

    for _key, value in G.nodes[extgrid_node].items():
        if isinstance(value, dict) and required.issubset(set(value.keys())):
            return value

    return None

def find_transformer_impedance_at_extgrid(
    G: nx.MultiGraph,
    extgrid_node: str,
    Un_lv: float = 110.0,
) -> tuple[complex, complex]:
    """
    Find transformer edge connected to the external-grid bus and return
    positive-/zero-sequence transformer impedance.
    """
    for _u, _v, key, attr in G.edges(extgrid_node, keys=True, data=True):
        pdict = attr.get(f"{key}_param_dict")

        if pdict and "uktr" in pdict and "uk0tr" in pdict:
            return transformer_impedance_from_edge_param(pdict, Un_l_kV=Un_lv)

    raise RuntimeError(
        f"No transformer impedance edge found connected to {extgrid_node!r}"
    )

def referred_source_at_110(
    G: nx.MultiGraph,
    extgrid_node: str,
    Un_hv: float = 380.0,
    Un_lv: float = 110.0,
    allow_transformer_only_fallback: bool = True,
) -> tuple[complex, complex]:
    Z1_tr, Z0_tr = find_transformer_impedance_at_extgrid(
        G=G,
        extgrid_node=extgrid_node,
        Un_lv=Un_lv,
    )

    extgrid_params = find_extgrid_param_dict_on_node(G, extgrid_node)

    if extgrid_params is None:
        if not allow_transformer_only_fallback:
            raise KeyError(
                f"No external-grid source parameter dictionary found on node "
                f"{extgrid_node!r}. Needed keys: snss, rntxn, x0tx1, r0tx0."
            )

        print(
            f"WARNING: No external-grid source parameters found on {extgrid_node!r}. "
            f"Using transformer-only source impedance fallback."
        )
        return Z1_tr, Z0_tr

    Z1_src_hv, Z0_src_hv = extgrid_source_impedance(extgrid_params, Un_hv)

    ratio = (Un_lv / Un_hv) ** 2
    Z1_src = Z1_src_hv * ratio + Z1_tr
    Z0_src = Z0_src_hv * ratio + Z0_tr

    return Z1_src, Z0_src


def load_graph(graph_path: str) -> nx.MultiGraph:
    """Load the networkx MultiGraph from pickle."""
    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    if not isinstance(G, nx.MultiGraph):
        raise TypeError(f"Expected networkx.MultiGraph, got {type(G).__name__}")

    return G


def load_takagi_impedance_bank(
    graph_path: str,
    line_source_mapping: dict[str, tuple[str, str]] | None = None,
) -> dict[str, dict[str, complex]]:
    """
    Return source/line impedance inputs for each faulted line.

    Output:
        {
            "MainLn1-2A": {
                "Z1_line": complex,
                "Z0_line": complex,
                "Z1_src_local": complex,
                "Z0_src_local": complex,
                "Z1_src_remote": complex,
                "Z0_src_remote": complex,
            },
            ...
        }

    By default, this uses the same local/remote source approximation as the
    extraction script:
        local  -> 
        remote -> BusExtGrid2

    For a more exact meshed-grid Thevenin equivalent, this mapping/function
    would need to be replaced by a network-reduction calculation.
    """
    G = load_graph(graph_path)

    lines = line_impedances(G)

    if line_source_mapping is None:
        line_source_mapping = {
            "MainLn1-2A": ("BusExtGrid1", "BusExtGrid2"),
            "MainLn1-2B": ("BusExtGrid1", "BusExtGrid2"),
            "MainLn2-3A": ("BusExtGrid1", "BusExtGrid2"),
            "MainLn2-3B": ("BusExtGrid1", "BusExtGrid2"),
        }

    out: dict[str, dict[str, complex]] = {}

    for line_name, (local_grid, remote_grid) in line_source_mapping.items():
        if line_name not in lines:
            raise KeyError(
                f"Line {line_name!r} not found in graph line impedances. "
                f"Available lines: {sorted(lines)}"
            )

        Z1_line, Z0_line = lines[line_name]

        Z1_src_local, Z0_src_local = referred_source_at_110(G, local_grid)
        Z1_src_remote, Z0_src_remote = referred_source_at_110(G, remote_grid)

        out[line_name] = {
            "Z1_line": Z1_line,
            "Z0_line": Z0_line,
            "Z1_src_local": Z1_src_local,
            "Z0_src_local": Z0_src_local,
            "Z1_src_remote": Z1_src_remote,
            "Z0_src_remote": Z0_src_remote,
        }

    return out


def print_takagi_impedance_bank(bank: dict[str, dict[str, complex]]) -> None:
    """Small debug printer."""
    for line_name, values in bank.items():
        print(f"\n{line_name}")
        for key, value in values.items():
            print(f"  {key:15s}: {value:.4f}")
