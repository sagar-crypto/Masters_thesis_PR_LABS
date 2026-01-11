from __future__ import annotations

import os, json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# -------------------------
# Core helpers
# -------------------------
def estimate_fs(t: np.ndarray) -> float:
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    dt = dt[dt > 0]
    if len(dt) == 0:
        raise ValueError("Cannot estimate fs.")
    return 1.0 / float(np.median(dt))


def dft_phasor_1cycle(x: np.ndarray) -> complex:
    N = len(x)
    n = np.arange(N, dtype=np.float64)
    w = np.exp(-1j * 2.0 * np.pi * n / N)
    return (2.0 / N) * np.sum(x.astype(np.float64) * w)


def compute_classical_distance(Z_app: complex, r1: float, x1: float, L_km: float) -> float:
    Z1 = complex(r1, x1)
    if abs(Z1) < 1e-12:
        return np.nan
    d = (abs(Z_app) / abs(Z1)) * L_km
    return float(np.clip(d, 0.0, L_km))


def k0_from_line(r1: float, x1: float, r0: float, x0: float) -> complex:
    Z1 = complex(r1, x1)
    Z0 = complex(r0, x0)
    if abs(Z1) < 1e-12:
        return 0j
    return (Z0 - Z1) / Z1


def symm_pos_seq(a: complex, b: complex, c: complex) -> complex:
    """Return positive-sequence component only (V1 or I1)."""
    alpha = np.exp(1j * 2.0 * np.pi / 3.0)
    a2 = alpha ** 2
    return (a + alpha * b + a2 * c) / 3.0


def _norm(s: str) -> str:
    return str(s).strip().lower().replace(" ", "").replace("_", "")


def select_case(sc_type: str, phase_select: str) -> str:
    st = _norm(sc_type)
    ph = _norm(phase_select)
    if ph in {"abc", "3ph", "3phase"}:
        ph = "abc"

    if st in {"3ph", "3phg", "3phase", "3phaseg"}:
        return "3ph"

    if st in {"1phg", "slg", "lg"}:
        return f"slg_{ph}" if ph in {"a", "b", "c"} else "3ph"

    if st in {"2ph", "ll"}:
        return f"ll_{ph}" if ph in {"ab", "bc", "ca"} else "3ph"

    if st in {"2phg", "llg"}:
        return f"llg_{ph}" if ph in {"ab", "bc", "ca"} else "3ph"

    return "3ph"


# -------------------------
# Column spec
# -------------------------
@dataclass
class ColumnSpec:
    time_index: int
    keep_indices: List[int]
    stack_order: List[str]
    group_name: str


def load_column_spec(json_path: str, group_name: str) -> ColumnSpec:
    with open(json_path, "r") as f:
        spec = json.load(f)
    g = spec["signal_groups"][group_name]
    keep_cols = g["keep_cols"]
    stack_order = g.get("stack_order", [])
    if not keep_cols:
        raise ValueError("keep_cols empty in JSON.")
    if stack_order and len(stack_order) != len(keep_cols):
        raise ValueError("stack_order length must match keep_cols length.")
    return ColumnSpec(
        time_index=0,
        keep_indices=list(range(1, 1 + len(keep_cols))),
        stack_order=stack_order if stack_order else [f"ch{i}" for i in range(len(keep_cols))],
        group_name=group_name,
    )


# -------------------------
# Event index
# -------------------------
@dataclass
class EventIndex:
    rep_ids: np.ndarray
    start_idx: np.ndarray
    end_idx: np.ndarray
    y: np.ndarray
    fs_est: np.ndarray
    r1: np.ndarray
    x1: np.ndarray
    r0: np.ndarray
    x0: np.ndarray
    L_km: np.ndarray
    sc_type: np.ndarray
    phase_select: np.ndarray
    valid: np.ndarray
    fault_r: np.ndarray


def build_event_index_from_npy(
    labels_csv: str,
    npy_dir: str,
    pre_ms: float,
    post_ms: float,
    line_prefix: str,
    time_col_index: int = 0,
) -> EventIndex:
    labels = pd.read_csv(labels_csv, sep=";").copy().sort_values("rep_id")

    rep_ids, s_idx, e_idx, y, fs_est = [], [], [], [], []
    r1, x1, r0, x0, L_km = [], [], [], [], []
    sc_type, phase_select,fault_r, valid = [], [], [], []

    for _, row in labels.iterrows():
        rid = int(row["rep_id"])
        path = os.path.join(npy_dir, f"replica_{rid}.npy")
        if not os.path.exists(path):
            continue

        arr = np.load(path, mmap_mode="r")
        t = np.asarray(arr[:, time_col_index], dtype=np.float64)

        try:
            fs = estimate_fs(t)
        except Exception:
            continue

        t_fault = float(row["t_evnt_start"])
        f_idx = int(np.argmin(np.abs(t - t_fault)))

        pre_samp = int(round((pre_ms / 1000.0) * fs))
        post_samp = int(round((post_ms / 1000.0) * fs))

        s = f_idx - pre_samp
        e = f_idx + post_samp

        ok = (s >= 0) and (e <= len(t)) and (e > s)

        rep_ids.append(rid)
        s_idx.append(s)
        e_idx.append(e)
        y.append(float(row["sc_location"]))
        fs_est.append(float(fs))

        L_km.append(float(row[f"{line_prefix}_length"]))
        r1.append(float(row[f"{line_prefix}_rline"]))
        x1.append(float(row[f"{line_prefix}_xline"]))
        r0.append(float(row[f"{line_prefix}_rline0"]))
        x0.append(float(row[f"{line_prefix}_xline0"]))

        sc_type.append(str(row["sc_type"]))
        phase_select.append(str(row["phase_select"]))
        fault_r.append(float(row["fault_resistance"]))
        valid.append(ok)

    return EventIndex(
        rep_ids=np.asarray(rep_ids, dtype=np.int32),
        start_idx=np.asarray(s_idx, dtype=np.int32),
        end_idx=np.asarray(e_idx, dtype=np.int32),
        y=np.asarray(y, dtype=np.float32),
        fs_est=np.asarray(fs_est, dtype=np.float32),
        r1=np.asarray(r1, dtype=np.float32),
        x1=np.asarray(x1, dtype=np.float32),
        r0=np.asarray(r0, dtype=np.float32),
        x0=np.asarray(x0, dtype=np.float32),
        L_km=np.asarray(L_km, dtype=np.float32),
        sc_type=np.asarray(sc_type, dtype=object),
        phase_select=np.asarray(phase_select, dtype=object),
        fault_r=np.asarray(fault_r, dtype=np.float32),
        valid=np.asarray(valid, dtype=bool),
    )


# -------------------------
# NPY cache
# -------------------------
class ReplicaCache:
    def __init__(self, npy_dir: str, max_items: int = 2):
        self.npy_dir = npy_dir
        self.max_items = max_items
        self._cache: Dict[int, np.ndarray] = {}
        self._order: List[int] = []

    def get(self, rep_id: int) -> np.ndarray:
        if rep_id in self._cache:
            self._order.remove(rep_id)
            self._order.append(rep_id)
            return self._cache[rep_id]
        path = os.path.join(self.npy_dir, f"replica_{rep_id}.npy")
        arr = np.load(path, mmap_mode="r")
        self._cache[rep_id] = arr
        self._order.append(rep_id)
        if len(self._order) > self.max_items:
            old = self._order.pop(0)
            self._cache.pop(old, None)
        return arr


# -------------------------
# Dataset
# -------------------------
class FaultWindowNpyDataset(Dataset):
    """
    Returns:
      X_win: (T,C) window normalized per setting
      y: scalar
      classic_pct: scalar baseline
      ko_feats: (6,) [Zpre_r,Zpre_i,Zpost_r,Zpost_i,dZ_r,dZ_i]
    """

    def __init__(
        self,
        event_index: EventIndex,
        npy_dir: str,
        colspec: ColumnSpec,
        pre_ms: float,
        f_nom: float = 50.0,
        window_cycles: int = 1,
        only_valid: bool = True,
        normalize: str = "train_global",  # "train_global" | "per_window" | "none"
        train_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        return_ko_feats: bool = True,
    ):
        self.cache = ReplicaCache(npy_dir=npy_dir, max_items=2)
        self.colspec = colspec
        self.pre_ms = pre_ms
        self.f_nom = f_nom
        self.window_cycles = window_cycles
        self.normalize = normalize
        self.train_stats = train_stats
        self.return_ko_feats = return_ko_feats

        if len(colspec.keep_indices) < 6:
            raise ValueError("Need 6 channels: [Va,Vb,Vc,Ia,Ib,Ic]")

        keep = event_index.valid if only_valid else np.ones_like(event_index.valid, bool)

        self.rep_ids = event_index.rep_ids[keep]
        self.start_idx = event_index.start_idx[keep]
        self.end_idx = event_index.end_idx[keep]
        self.y = event_index.y[keep]
        self.fs_est = event_index.fs_est[keep]
        self.r1 = event_index.r1[keep]
        self.x1 = event_index.x1[keep]
        self.r0 = event_index.r0[keep]
        self.x0 = event_index.x0[keep]
        self.L_km = event_index.L_km[keep]
        self.sc_type = event_index.sc_type[keep]
        self.phase_select = event_index.phase_select[keep]
        self.fault_r = event_index.fault_r[keep]

    def __len__(self) -> int:
        return int(len(self.y))

    def _norm_window(self, X: np.ndarray) -> np.ndarray:
        if self.normalize == "none":
            return X
        if self.normalize == "per_window":
            mu = X.mean(axis=0, keepdims=True)
            sd = X.std(axis=0, keepdims=True) + 1e-6
            return (X - mu) / sd
        if self.normalize == "train_global":
            if self.train_stats is None:
                mu = X.mean(axis=0, keepdims=True)
                sd = X.std(axis=0, keepdims=True) + 1e-6
                return (X - mu) / sd
            mu, sd = self.train_stats
            return (X - mu[None, :]) / (sd[None, :] + 1e-6)
        raise ValueError("normalize must be: train_global | per_window | none")

    def _compute_ko(self, X_raw: np.ndarray, fs: float, r1: float, x1: float, r0: float, x0: float, L: float,
                   sc_type: str, phase_select: str) -> Tuple[np.float32, np.ndarray]:
        # phasor slices
        pre_samples = int(round((self.pre_ms / 1000.0) * fs))
        spc = int(round(fs / self.f_nom))
        cyc = self.window_cycles * spc

        pre_end = pre_samples
        pre_start = pre_end - cyc
        post_start = pre_samples
        post_end = post_start + cyc
        if pre_start < 0 or post_end > X_raw.shape[0]:
            return np.float32(0.0), np.zeros((6,), np.float32)

        # channels [Va,Vb,Vc,Ia,Ib,Ic]
        Va_pre = dft_phasor_1cycle(X_raw[pre_start:pre_end, 0])
        Vb_pre = dft_phasor_1cycle(X_raw[pre_start:pre_end, 1])
        Vc_pre = dft_phasor_1cycle(X_raw[pre_start:pre_end, 2])
        Ia_pre = dft_phasor_1cycle(X_raw[pre_start:pre_end, 3])
        Ib_pre = dft_phasor_1cycle(X_raw[pre_start:pre_end, 4])
        Ic_pre = dft_phasor_1cycle(X_raw[pre_start:pre_end, 5])

        Va_po = dft_phasor_1cycle(X_raw[post_start:post_end, 0])
        Vb_po = dft_phasor_1cycle(X_raw[post_start:post_end, 1])
        Vc_po = dft_phasor_1cycle(X_raw[post_start:post_end, 2])
        Ia_po = dft_phasor_1cycle(X_raw[post_start:post_end, 3])
        Ib_po = dft_phasor_1cycle(X_raw[post_start:post_end, 4])
        Ic_po = dft_phasor_1cycle(X_raw[post_start:post_end, 5])

        k0 = k0_from_line(r1, x1, r0, x0)
        case = select_case(sc_type, phase_select)

        eps = 1e-9
        I0_pre = (Ia_pre + Ib_pre + Ic_pre) / 3.0
        I0_po = (Ia_po + Ib_po + Ic_po) / 3.0

        def Z(v, i):
            return v / (i + eps)

        if case == "3ph":
            V1_pre = symm_pos_seq(Va_pre, Vb_pre, Vc_pre)
            I1_pre = symm_pos_seq(Ia_pre, Ib_pre, Ic_pre)
            V1_po = symm_pos_seq(Va_po, Vb_po, Vc_po)
            I1_po = symm_pos_seq(Ia_po, Ib_po, Ic_po)
            Zpre = Z(V1_pre, I1_pre)
            Zpo = Z(V1_po, I1_po)

        elif case == "slg_a":
            Zpre = Va_pre / (Ia_pre + k0 * I0_pre + eps)
            Zpo  = Va_po  / (Ia_po  + k0 * I0_po  + eps)
        elif case == "slg_b":
            Zpre = Vb_pre / (Ib_pre + k0 * I0_pre + eps)
            Zpo  = Vb_po  / (Ib_po  + k0 * I0_po  + eps)
        elif case == "slg_c":
            Zpre = Vc_pre / (Ic_pre + k0 * I0_pre + eps)
            Zpo  = Vc_po  / (Ic_po  + k0 * I0_po  + eps)

        elif case == "ll_ab":
            Zpre = (Va_pre - Vb_pre) / ((Ia_pre - Ib_pre) + eps)
            Zpo  = (Va_po  - Vb_po)  / ((Ia_po  - Ib_po)  + eps)
        elif case == "ll_bc":
            Zpre = (Vb_pre - Vc_pre) / ((Ib_pre - Ic_pre) + eps)
            Zpo  = (Vb_po  - Vc_po)  / ((Ib_po  - Ic_po)  + eps)
        elif case == "ll_ca":
            Zpre = (Vc_pre - Va_pre) / ((Ic_pre - Ia_pre) + eps)
            Zpo  = (Vc_po  - Va_po)  / ((Ic_po  - Ia_po)  + eps)

        elif case == "llg_ab":
            Zpre = (Va_pre - Vb_pre) / ((Ia_pre - Ib_pre) + k0 * I0_pre + eps)
            Zpo  = (Va_po  - Vb_po)  / ((Ia_po  - Ib_po)  + k0 * I0_po  + eps)
        elif case == "llg_bc":
            Zpre = (Vb_pre - Vc_pre) / ((Ib_pre - Ic_pre) + k0 * I0_pre + eps)
            Zpo  = (Vb_po  - Vc_po)  / ((Ib_po  - Ic_po)  + k0 * I0_po  + eps)
        else:  # llg_ca
            Zpre = (Vc_pre - Va_pre) / ((Ic_pre - Ia_pre) + k0 * I0_pre + eps)
            Zpo  = (Vc_po  - Va_po)  / ((Ic_po  - Ia_po)  + k0 * I0_po  + eps)

        dZ = Zpo - Zpre
        d_km = compute_classical_distance(Zpo, r1, x1, L)
        classic_pct = np.float32(100.0 * d_km / max(L, 1e-12))

        feats = np.array([np.real(Zpre), np.imag(Zpre), np.real(Zpo), np.imag(Zpo), np.real(dZ), np.imag(dZ)],
                         dtype=np.float32)
        return classic_pct, feats

    def __getitem__(self, idx: int):
        rid = int(self.rep_ids[idx])
        s = int(self.start_idx[idx])
        e = int(self.end_idx[idx])

        arr = self.cache.get(rid)
        win = np.asarray(arr[s:e, :], dtype=np.float32)

        X_raw = win[:, self.colspec.keep_indices].astype(np.float32)   # (T,C)
        X_win = self._norm_window(X_raw.copy())                        # (T,C) normalized for NN

        y = np.float32(self.y[idx])

        classic_pct, ko_feats = self._compute_ko(
            X_raw=X_raw,
            fs=float(self.fs_est[idx]),
            r1=float(self.r1[idx]),
            x1=float(self.x1[idx]),
            r0=float(self.r0[idx]),
            x0=float(self.x0[idx]),
            L=float(self.L_km[idx]),
            sc_type=str(self.sc_type[idx]),
            phase_select=str(self.phase_select[idx]),
        )

        # --- NEW: build context vector (float32) ---
        # context = [classic_pct, ko_feats(6), r1,x1,r0,x0,L_km,(fault_r)]
        ctx_parts = [
            np.array([classic_pct], dtype=np.float32),         # (1,)
            ko_feats.astype(np.float32),                       # (6,)
            np.array([
                float(self.r1[idx]), float(self.x1[idx]),
                float(self.r0[idx]), float(self.x0[idx]),
                float(self.L_km[idx]),
            ], dtype=np.float32),                               # (5,)
        ]

        # Optional: include fault resistance if you added it to EventIndex/Dataset
        if hasattr(self, "fault_r"):
            ctx_parts.append(np.array([float(self.fault_r[idx])], dtype=np.float32))  # (1,)

        context = np.concatenate(ctx_parts, axis=0).astype(np.float32)  # (12) or (13)

        if self.return_ko_feats:
            return (
                torch.from_numpy(X_win),                        # (T,C)
                torch.tensor(y, dtype=torch.float32),           # ()
                torch.tensor(classic_pct, dtype=torch.float32), # ()
                torch.from_numpy(ko_feats),                     # (6,)
                torch.from_numpy(context),                      # (12) or (13,)
            )

        return (
            torch.from_numpy(X_win),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(classic_pct, dtype=torch.float32),
            torch.from_numpy(context),
        )



# -------------------------
# Train stats (optional)
# -------------------------
def compute_train_stats(
    dataset,
    indices: np.ndarray,
    max_items: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel mean/std over a subset of windows.

    IMPORTANT:
    - dataset must be created with normalize="none"
    - indices must refer to the dataset AFTER filtering (only_valid)
    """
    if indices is None or len(indices) == 0:
        raise ValueError("compute_train_stats: received empty indices.")

    if max_items is not None:
        indices = indices[:max_items]

    sums = None
    sqs = None
    count = 0

    for i in indices:
        X = dataset[int(i)][0]   # always X_win
        Xn = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else np.asarray(X)

        if sums is None:
            C = Xn.shape[1]
            sums = np.zeros((C,), dtype=np.float64)
            sqs = np.zeros((C,), dtype=np.float64)

        sums += Xn.sum(axis=0)
        sqs += (Xn ** 2).sum(axis=0)
        count += Xn.shape[0]

    if sums is None or sqs is None or count == 0:
        raise RuntimeError(
            "compute_train_stats: no samples accumulated. "
            "Likely causes: empty indices, all events filtered out, or max_items=0."
        )

    mu = sums / count
    var = sqs / count - mu ** 2
    sigma = np.sqrt(np.maximum(var, 1e-12))
    return mu.astype(np.float32), sigma.astype(np.float32)

