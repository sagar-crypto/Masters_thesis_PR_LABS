import numpy as np
from sklearn.metrics import mean_absolute_error
import torch
from sklearn.linear_model import LinearRegression

SCALE = 100.0

def analyze_plateau(model, loader, device, ctx_fault_type_idx=None, eps=1e-6, post_frac=0.25):
    """
    Prints MAE slices:
      1) by true distance bins
      2) by fault type (if ctx_fault_type_idx provided)
      3) by near-zero post-fault (based on waveform magnitude)
    Args:
      ctx_fault_type_idx: list of indices in ctx corresponding to one-hot fault type, e.g. [0,1,2,3]
                          If None, skips fault type slicing.
      eps: threshold for "near zero" magnitude
      post_frac: fraction of window treated as post-fault (last 25% by default)
    """
    model.eval()

    y_list, yhat_list, ft_list, nearzero_list = [], [], [], []

    with torch.no_grad():
        for xb, yb, cb, _ko, ctx in loader:
            xb = xb.to(device, non_blocking=True)   # [B, C, T] or [B, T, C]
            yb = yb.to(device, non_blocking=True)   # [%]
            cb = cb.to(device, non_blocking=True)   # [%]
            ctx = ctx.to(device, non_blocking=True)

            # ---- prediction in normalized space ----
            yb_n = (yb / SCALE).view(-1, 1)        # (B,1)
            cb_n = (cb / SCALE).view(-1, 1)        # (B,1)

            delta_n = model(xb, ctx).view(-1, 1)   # (B,1)
            yhat_n = cb_n + delta_n                # (B,1)
            yhat_n = torch.clamp(yhat_n, 0.0, 1.0)

            # store in % for MAE reporting
            y = (yb_n * SCALE).detach().cpu().numpy().reshape(-1)
            yhat = (yhat_n * SCALE).detach().cpu().numpy().reshape(-1)
            y_list.append(y)
            yhat_list.append(yhat)

            # ---- fault type extraction (optional) ----
            if ctx_fault_type_idx is not None:
                ctx_cpu = ctx.detach().cpu().numpy()
                ft_onehot = ctx_cpu[:, ctx_fault_type_idx]   # [B, K]
                ft = np.argmax(ft_onehot, axis=1)            # 0..K-1
                ft_list.append(ft)

            # ---- near-zero post-fault detection ----
            # We compute magnitude over last post_frac of time dimension.
            x = xb.detach().cpu().numpy()

            # Support both [B, C, T] and [B, T, C]
            if x.ndim != 3:
                raise ValueError(f"Expected xb to be 3D, got shape {x.shape}")

            if x.shape[1] < x.shape[2]:  # likely [B, C, T]
                B, C, T = x.shape
                t0 = int((1.0 - post_frac) * T)
                post = x[:, :, t0:]                      # [B, C, Tpost]
                mag = np.mean(np.abs(post), axis=(1, 2)) # [B]
            else:  # likely [B, T, C]
                B, T, C = x.shape
                t0 = int((1.0 - post_frac) * T)
                post = x[:, t0:, :]                      # [B, Tpost, C]
                mag = np.mean(np.abs(post), axis=(1, 2)) # [B]

            nearzero = (mag < eps).astype(np.int32)      # 1 if near-zero
            nearzero_list.append(nearzero)

    y = np.concatenate(y_list)
    yhat = np.concatenate(yhat_list)
    abs_err = np.abs(yhat - y)

    print("\n=== Overall ===")
    print(f"MAE:  {mean_absolute_error(y, yhat):.2f} %")
    print(f"P90:  {np.percentile(abs_err, 90):.2f} %")
    print(f"P95:  {np.percentile(abs_err, 95):.2f} %")

    # 1) distance bins
    print("\n=== MAE by True Distance Bin ===")
    bins = [(0,20),(20,40),(40,60),(60,80),(80,100)]
    for lo, hi in bins:
        m = (y >= lo) & (y < hi) if hi < 100 else (y >= lo) & (y <= hi)
        if m.sum() == 0:
            continue
        print(f"{lo:02d}-{hi:03d}% | n={m.sum():4d} | MAE={mean_absolute_error(y[m], yhat[m]):.2f} %")

    # 2) fault type
    if ctx_fault_type_idx is not None:
        ft = np.concatenate(ft_list)
        print("\n=== MAE by Fault Type (from ctx one-hot) ===")
        for k in np.unique(ft):
            m = (ft == k)
            print(f"type={k} | n={m.sum():4d} | MAE={mean_absolute_error(y[m], yhat[m]):.2f} %")

    # 3) near-zero post-fault
    nz = np.concatenate(nearzero_list).astype(bool)
    print("\n=== MAE by Near-Zero Post-Fault ===")
    if nz.sum() > 0:
        print(f"near-zero=1 | n={nz.sum():4d} | MAE={mean_absolute_error(y[nz], yhat[nz]):.2f} %")
    print(f"near-zero=0 | n={(~nz).sum():4d} | MAE={mean_absolute_error(y[~nz], yhat[~nz]):.2f} %")



def fit_classic_calibration(train_loader, device):
    """
    Fit y ≈ a*cb + b on TRAIN only.
    Returns (a, b).
    """
    cbs, ys = [], []
    for _, yb, cb, _, _ in train_loader:
        # yb, cb are in % already (based on your prints)
        ys.append(yb.detach().cpu().numpy().reshape(-1, 1))
        cbs.append(cb.detach().cpu().numpy().reshape(-1, 1))

    y = np.vstack(ys)      # shape [N,1]
    cb = np.vstack(cbs)    # shape [N,1]

    reg = LinearRegression(fit_intercept=True)
    reg.fit(cb, y)

    a = float(reg.coef_.ravel()[0])
    b = float(reg.intercept_)

    return a, b


def eval_classic_baselines(test_loader, a, b):
    ys, cb_raws, cb_cals = [], [], []

    for _, yb, cb, _, _ in test_loader:
        y = yb.detach().cpu().numpy().reshape(-1)
        cb_raw = cb.detach().cpu().numpy().reshape(-1)
        cb_cal = a * cb_raw + b

        ys.append(y)
        cb_raws.append(cb_raw)
        cb_cals.append(cb_cal)

    y = np.concatenate(ys)
    cb_raw = np.concatenate(cb_raws)
    cb_cal = np.concatenate(cb_cals)

    def print_bins(name, pred):
        print(f"\n=== {name} ===")
        print(f"Overall MAE: {mean_absolute_error(y, pred):.2f} %")
        bins = [(0,20),(20,40),(40,60),(60,80),(80,100)]
        for lo, hi in bins:
            m = (y >= lo) & (y < hi) if hi < 100 else (y >= lo) & (y <= hi)
            if m.sum() == 0: 
                continue
            print(f"{lo:02d}-{hi:03d}% | n={m.sum():4d} | MAE={mean_absolute_error(y[m], pred[m]):.2f} %")

    print_bins("Classic RAW (cb)", cb_raw)
    print_bins("Classic CALIBRATED (a*cb+b)", cb_cal)



def eval_flip_baseline(test_loader):
    ys, cb_raws = [], []
    for _, yb, cb, _, _ in test_loader:
        y = yb.detach().cpu().numpy().reshape(-1)
        cb_raw = cb.detach().cpu().numpy().reshape(-1)
        ys.append(y)
        cb_raws.append(cb_raw)

    y = np.concatenate(ys)
    cb = np.concatenate(cb_raws)

    cb_flip = 100.0 - cb

    print("\n=== Classic FLIPPED (100 - cb) ===")
    print(f"Overall MAE: {mean_absolute_error(y, cb_flip):.2f} %")
    bins = [(0,20),(20,40),(40,60),(60,80),(80,100)]
    for lo, hi in bins:
        m = (y >= lo) & (y < hi) if hi < 100 else (y >= lo) & (y <= hi)
        if m.sum() == 0: 
            continue
        print(f"{lo:02d}-{hi:03d}% | n={m.sum():4d} | MAE={mean_absolute_error(y[m], cb_flip[m]):.2f} %")


def eval_best_of_two_baseline(test_loader):
    ys, cbs = [], []
    for _, yb, cb, _, _ in test_loader:
        y = yb.detach().cpu().numpy().reshape(-1)
        c = cb.detach().cpu().numpy().reshape(-1)
        ys.append(y); cbs.append(c)

    y = np.concatenate(ys)
    cb = np.concatenate(cbs)
    cb_flip = 100.0 - cb

    best = np.where(np.abs(y - cb) <= np.abs(y - cb_flip), cb, cb_flip)

    print("\n=== Classic BEST-OF-TWO (min(|y-cb|, |y-(100-cb)|)) ===")
    print(f"Overall MAE: {mean_absolute_error(y, best):.2f} %")

    bins = [(0,20),(20,40),(40,60),(60,80),(80,100)]
    for lo, hi in bins:
        m = (y >= lo) & (y < hi) if hi < 100 else (y >= lo) & (y <= hi)
        if m.sum() == 0:
            continue
        print(f"{lo:02d}-{hi:03d}% | n={m.sum():4d} | MAE={mean_absolute_error(y[m], best[m]):.2f} %")
