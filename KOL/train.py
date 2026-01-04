# train.py
from __future__ import annotations
import os
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from config import LABELS_CSV, CACHE_FILE_PATH, COLUMN_SPEC_JSON
from data_processing import (
    load_column_spec,
    build_event_index_from_npy,
    FaultWindowNpyDataset,
    compute_train_stats,
)
from model import WaveDeltaCNN


# -------------------------
# TRAIN CONFIG (keep minimal here)
# -------------------------
GROUP_NAME   = "Bus1Line_1_2_a"   # name inside your JSON under signal_groups
LINE_PREFIX  = "line_1_2_a"       # labels.csv column prefix for rline/xline/length

PRE_MS  = 40.0
POST_MS = 40.0

F_NOM = 50.0
WINDOW_CYCLES = 1

EPOCHS = 150
BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 1e-4

USE_SMOOTHL1 = True
DROPOUT = 0.2

NUM_WORKERS = 4
PIN_MEMORY = True

SUBSAMPLE_FRAC = 0.10      # 10% of valid events
SUBSAMPLE_MAX  = 500       # hard cap (None to disable)
SMOKE_ONLY     = False      # set False to train on full data

EVAL_EPOCHS = {1, 5, 10, 20, 30, 50}


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    for xb, yb, cb, _ko in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        cb = cb.to(device, non_blocking=True)

        delta = model(xb, cb)
        y_hat = cb + delta

        preds.append(y_hat.detach().cpu().numpy())
        trues.append(yb.detach().cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    return float(mean_absolute_error(trues, preds))


def train_one_seed(seed: int = 42) -> dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Load column spec (JSON)
    # -------------------------
    colspec = load_column_spec(COLUMN_SPEC_JSON, group_name=GROUP_NAME)

    # -------------------------
    # Build bitmap event index from NPY memmap
    # -------------------------
    index = build_event_index_from_npy(
        labels_csv=LABELS_CSV,
        npy_dir=CACHE_FILE_PATH,
        pre_ms=PRE_MS,
        post_ms=POST_MS,
        line_prefix=LINE_PREFIX,
        time_col_index=colspec.time_index,
    )

    # Filter valid events (bitmap)
    valid_ids = np.where(index.valid)[0]
    if len(valid_ids) < 10:
        raise RuntimeError(f"Too few valid events: {len(valid_ids)}. Check PRE/POST or file availability.")
    

    # ---- SMOKE TEST SUBSAMPLE ----
    if SMOKE_ONLY:
        rng = np.random.default_rng(seed)
        n = len(valid_ids)
        k = max(10, int(round(n * SUBSAMPLE_FRAC)))
        if SUBSAMPLE_MAX is not None:
            k = min(k, int(SUBSAMPLE_MAX))
        valid_ids = rng.choice(valid_ids, size=k, replace=False)
        valid_ids = np.sort(valid_ids)
        print(f"[seed={seed}] SMOKE_ONLY enabled -> using {len(valid_ids)} / {n} valid events")

    # -------------------------
    # Split
    # -------------------------
    tr_ids, te_ids = train_test_split(valid_ids, test_size=0.2, random_state=seed)

    # -------------------------
    # Build dataset (RAW) to compute train normalization stats
    # -------------------------
    ds_raw = FaultWindowNpyDataset(
        event_index=index,
        npy_dir=CACHE_FILE_PATH,
        colspec=colspec,
        pre_ms=PRE_MS,
        f_nom=F_NOM,
        window_cycles=WINDOW_CYCLES,
        only_valid=False,           # we index explicitly by ids
        normalize="none",           # IMPORTANT for stats
        train_stats=None,
        return_ko_feats=True,
        ko_use_phase="a",
    )

    mu, sigma = compute_train_stats(ds_raw, indices=tr_ids, max_items=None)

    # -------------------------
    # Build dataset (normalized) for training
    # -------------------------
    ds = FaultWindowNpyDataset(
        event_index=index,
        npy_dir=CACHE_FILE_PATH,
        colspec=colspec,
        pre_ms=PRE_MS,
        f_nom=F_NOM,
        window_cycles=WINDOW_CYCLES,
        only_valid=False,
        normalize="train_global",
        train_stats=(mu, sigma),
        return_ko_feats=True,
        ko_use_phase="a",
    )

    train_ds = Subset(ds, tr_ids.tolist())
    test_ds  = Subset(ds, te_ids.tolist())

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=128, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False
    )

    # -------------------------
    # Classical baseline MAE (from classic_pct in dataset)
    # -------------------------
    # grab classic_pct quickly from test set
    y_true = []
    y_base = []
    for xb, yb, cb, _ko in test_loader:
        y_true.append(yb.numpy())
        y_base.append(cb.numpy())
    y_true = np.concatenate(y_true)
    y_base = np.concatenate(y_base)
    mae_classic = float(mean_absolute_error(y_true, y_base))

    print(f"[seed={seed}] valid={len(valid_ids)} train={len(tr_ids)} test={len(te_ids)} | Classical MAE: {mae_classic:.2f} %")

    # -------------------------
    # Model
    # -------------------------
    n_channels = len(colspec.keep_indices)
    model = WaveDeltaCNN(n_channels=n_channels, dropout=DROPOUT).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.SmoothL1Loss(beta=1.0) if USE_SMOOTHL1 else nn.L1Loss()

    best_mae = float("inf")
    best_epoch = None
    best_state = None

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb, cb, _ko in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            cb = cb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            delta = model(xb, cb)
            y_hat = cb + delta
            loss = loss_fn(y_hat, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if epoch in EVAL_EPOCHS:
            mae = evaluate(model, test_loader, device)
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"[seed={seed}] epoch {epoch:02d} | Test MAE: {mae:.2f} % | Best: {best_mae:.2f} % (ep {best_epoch})")

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "seed": seed,
        "n_valid": int(len(valid_ids)),
        "mae_classic": mae_classic,
        "best_mae_kol": float(best_mae),
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
    }


def main():
    # single seed (default)
    res = train_one_seed(seed=42)
    print("\n--- FINAL ---")
    print(f"Classical MAE: {res['mae_classic']:.2f} %")
    print(f"KOL MAE (best): {res['best_mae_kol']:.2f} % @ epoch {res['best_epoch']}")

    # Optional: multi-seed robustness
    # seeds = [0, 1, 2, 3, 4]
    # results = [train_one_seed(s) for s in seeds]
    # kol = np.array([r["best_mae_kol"] for r in results], dtype=np.float32)
    # print(f"\nKOL over seeds: mean={kol.mean():.2f} std={kol.std():.2f} | vals={kol}")


if __name__ == "__main__":
    main()
