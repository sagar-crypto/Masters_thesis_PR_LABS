from __future__ import annotations

import logging
import os
from typing import Optional

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from psp_helper.config import MainConfig
from psp_helper.utils.logging import get_logger
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

import dl_psp.data.labels as L
import wandb
from dl_psp.data.data_utils import load_windowed_dataset
from dl_psp.data.features import maybe_filter_features
from dl_psp.data.filters import (
    build_valid_row_indices,
    build_valid_row_indices_hv_double_line_90kv,
)
from dl_psp.data.targets import extract_target
from dl_psp.data.task_spec import get_task_spec, infer_task_type_from_spec
from dl_psp.models.model_utils import create_model_from_name, get_device
from dl_psp.utils.eval_utils import evaluate, predict_on_loader
from dl_psp.utils.run_utils import (
    get_env_info,
    infer_input_dims,
    save_checkpoint,
    save_fold_predictions,
    save_fold_splits,
    set_seed,
    set_torch_perf_flags,
)
from dl_psp.utils.subgroup_metrics import maybe_run_subgroup_analysis
from dl_psp.utils.train_utils import make_loaders, train_best_on_val
from dl_psp.utils.tuning_utils import tune_lr_wd_on_single_fold

logger = get_logger(__name__)


# =============================================================================
# Helpers mirrored from the benchmark runner
# =============================================================================

def split_train_val_from_train_pool(
    groups_used: pd.Series,
    train_pool_idx: np.ndarray,
    val_size: float = 0.2,
    split_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    pool_groups = (
        groups_used.iloc[train_pool_idx]
        if hasattr(groups_used, "iloc")
        else groups_used[train_pool_idx]
    )
    uniq = np.unique(pool_groups)
    rng = np.random.default_rng(split_seed)
    rng.shuffle(uniq)

    n_val_groups = max(1, int(round(val_size * len(uniq))))
    val_groups = set(uniq[:n_val_groups])

    idx_val = [
        i
        for i in train_pool_idx
        if (groups_used.iloc[i] if hasattr(groups_used, "iloc") else groups_used[i])
        in val_groups
    ]
    idx_val_set = set(idx_val)
    idx_train = [i for i in train_pool_idx if i not in idx_val_set]
    return np.array(idx_train, dtype=int), np.array(idx_val, dtype=int)


def canonicalize_multiclass_encoding(y_all: np.ndarray, task_type: str):
    if task_type != "multiclass":
        return y_all, {}

    unique_val = np.unique(y_all)

    if unique_val.dtype.kind in {"U", "S", "O"}:
        sorted_classes = sorted([str(x) for x in unique_val])
        canonical_mapping = {c: i for i, c in enumerate(sorted_classes)}
        y_encoded = np.array([canonical_mapping[str(v)] for v in y_all], dtype=np.int64)
    else:
        sorted_classes = sorted([int(x) for x in unique_val])
        canonical_mapping = {int(c): i for i, c in enumerate(sorted_classes)}
        y_encoded = np.array([canonical_mapping[int(v)] for v in y_all], dtype=np.int64)

    return y_encoded, canonical_mapping


def build_cv_splits_stratified(
    y_all: np.ndarray,
    groups_np: np.ndarray,
    task_type: str,
    n_splits: int,
    seed: int,
):
    if task_type == "multiclass":
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        return list(splitter.split(np.zeros(len(y_all)), y_all, groups_np))

    splitter = GroupKFold(n_splits=n_splits)
    return list(splitter.split(np.zeros(len(y_all)), y_all, groups_np))


def select_best_lr_wd(
    config: MainConfig,
    X_used,
    feature_indices_for_ds: Optional[list[int]],
    task_type: str,
    criterion,
    primary_name: str,
    higher_is_better: bool,
    valid_row_idx,
    labels_df_used: pd.DataFrame,
    y_all: np.ndarray,
    out_dim: int,
    F_eff: int,
    flat_dim: int,
    seed: int,
    n_splits: int,
):
    eval_only = bool(getattr(config.training, "eval_only", False))
    resave_eval_only = bool(getattr(config.training, "resave_eval_only", False))
    best_lr = float(config.training.learning_rate)
    best_wd = float(config.training.weight_decay)

    do_tune = bool(getattr(config.training, "tune_lr_wd", False))
    if do_tune and not eval_only:
        tune_cache_dir = os.path.join(
            str(getattr(config.training, "out_dir", "outputs")),
            "tuning_cache",
        )
        os.makedirs(tune_cache_dir, exist_ok=True)

        tune_lrs = list(getattr(config.training, "tune_lrs", [1e-4, 3e-4, 1e-3]))
        tune_wds = list(getattr(config.training, "tune_wds", [0.0, 1e-5, 1e-4]))

        tune_max_epochs_cfg = getattr(config.training, "tune_max_epochs", None)
        tune_max_epochs = 10 if tune_max_epochs_cfg is None else int(tune_max_epochs_cfg)

        tune_subsample_ratio = getattr(config.training, "tune_subsample_ratio", None)

        best_lr, best_wd, tune_info = tune_lr_wd_on_single_fold(
            config=config,
            X_used=X_used,
            y_all=y_all,
            task_type=task_type,
            feature_indices_for_ds=feature_indices_for_ds,
            valid_row_idx=valid_row_idx,
            labels_df_used=labels_df_used,
            fold_idx=0,
            seed=seed,
            F_eff=F_eff,
            flat_dim=flat_dim,
            out_dim=out_dim,
            criterion=criterion,
            primary_name=primary_name,
            higher_is_better=higher_is_better,
            lrs=tune_lrs,
            wds=tune_wds,
            cache_dir=tune_cache_dir,
            subsample_ratio=tune_subsample_ratio,
            max_epochs=tune_max_epochs,
            n_splits=int(n_splits),
        )
        logger.info(
            "Tuning selected lr=%g wd=%g | info=%s",
            best_lr,
            best_wd,
            tune_info,
        )

    return best_lr, best_wd, eval_only, resave_eval_only


def setup_wandb_logging(
    config: MainConfig,
    target_label: str,
    task_type: str,
    labels_df_used: pd.DataFrame,
    groups_used: pd.Series | np.ndarray,
    F_eff: int,
    model_name: str,
    window_s: float,
    window_ms: int,
    seed: int,
    env_info: dict,
):
    wb_run = None
    if bool(config.tracking.use_wandb):
        wb_run = wandb.init(
            project=config.tracking.project,
            entity=config.tracking.entity,
            mode=config.tracking.mode,
            name=f"{config.dataset.topology}__{target_label}__{model_name}__W{window_ms}ms",
            config={
                "topology": str(config.dataset.topology),
                "target": target_label,
                "task_type": task_type,
                "model": model_name,
                "n_features": int(F_eff),
                "window_length_s": float(window_s),
                "window_length_ms": int(window_ms),
                "batch_size": int(config.training.batch_size),
                "lr": float(config.training.learning_rate),
                "weight_decay": float(config.training.weight_decay),
                "epochs": int(config.training.epochs),
                "split_seed": int(config.training.split_seed),
                "n_seeds": 1,
                "git_commit": env_info.get("git_commit"),
                "git_status": env_info.get("git_status"),
                "python_version": env_info["python_version"].split()[0],
                "torch_version": env_info["torch_version"],
                "numpy_version": env_info["numpy_version"],
                "hostname": env_info["hostname"],
            },
        )

    if wb_run is not None:
        wb_run.summary["data/n_samples_used"] = int(len(labels_df_used))
        n_groups = (
            int(groups_used.nunique())
            if isinstance(groups_used, pd.Series)
            else len(np.unique(groups_used))
        )
        wb_run.summary["data/n_groups_used"] = int(n_groups)
        wb_run.summary["protocol/split_seed"] = int(config.training.split_seed)
        wb_run.summary["protocol/training_seeds"] = [int(seed)]
        wb_run.summary["env/timestamp"] = env_info["timestamp"]
        wb_run.summary["env/platform"] = env_info["platform"]
        wb_run.summary["env/cuda_available"] = env_info["cuda_available"]
        if env_info["cuda_available"]:
            wb_run.summary["env/cuda_version"] = env_info["cuda_version"]

    return wb_run


def validate_checkpoint_metadata(
    ckpt: dict,
    expected_topology: str,
    expected_target: str,
    expected_model: str,
    expected_window_s: float,
) -> None:
    meta = ckpt.get("meta", {})
    config_dict = ckpt.get("config", {})

    ckpt_topology = meta.get("topology") or config_dict.get("dataset", {}).get("topology")
    ckpt_target = meta.get("target_label") or config_dict.get("training", {}).get("target_label")
    ckpt_model = meta.get("model_name") or config_dict.get("model", {}).get("model_name")
    ckpt_window = meta.get("window_length_s")

    mismatches = []
    if ckpt_topology and str(ckpt_topology) != str(expected_topology):
        mismatches.append(f"topology: checkpoint={ckpt_topology}, expected={expected_topology}")
    if ckpt_target and str(ckpt_target) != str(expected_target):
        mismatches.append(f"target: checkpoint={ckpt_target}, expected={expected_target}")
    if ckpt_model and str(ckpt_model) != str(expected_model):
        mismatches.append(f"model: checkpoint={ckpt_model}, expected={expected_model}")
    if ckpt_window and abs(float(ckpt_window) - float(expected_window_s)) > 1e-6:
        mismatches.append(
            f"window_length: checkpoint={ckpt_window}, expected={expected_window_s}"
        )

    if mismatches:
        logger.warning(
            "Checkpoint metadata mismatch detected:\n  %s",
            "\n  ".join(mismatches),
        )


# =============================================================================
# KOL operator feature loading
# =============================================================================

def load_operator_features_if_enabled(
    config: MainConfig,
    labels_df_used: pd.DataFrame,
) -> Optional[np.ndarray]:
    use_ops = bool(getattr(config.training, "use_operator_features", False))
    if not use_ops:
        logger.info("Operator features disabled. Running plain DL baseline.")
        return None

    operator_path = getattr(config.training, "operator_features_path", None)
    if operator_path is None:
        raise ValueError(
            "use_operator_features=true but training.operator_features_path is not set."
        )

    logger.info("Loading operator features from: %s", operator_path)

    if str(operator_path).endswith(".parquet"):
        ops_df = pd.read_parquet(operator_path)
    else:
        ops_df = pd.read_csv(operator_path)

    merge_keys = ["sample_id"]
    if "window_idx" in labels_df_used.columns and "window_idx" in ops_df.columns:
        merge_keys = ["sample_id", "window_idx"]

    merged = labels_df_used.reset_index(drop=True).merge(
        ops_df,
        on=merge_keys,
        how="left",
        suffixes=("", "_op"),
    )

    if len(merged) != len(labels_df_used):
        raise RuntimeError("Operator feature merge changed row count unexpectedly.")

    op_cols = list(
        getattr(
            config.training,
            "operator_feature_columns",
            [
                "d_phys_real_pct",
                "d_phys_abs_pct",
                "ratio_real",
                "ratio_abs",
                "z_app_real",
                "z_app_imag",
            ],
        )
    )

    missing_cols = [c for c in op_cols if c not in merged.columns]
    if missing_cols:
        raise ValueError(f"Missing operator feature columns after merge: {missing_cols}")

    Z_ops = merged[op_cols].astype(np.float32).to_numpy()
    missing_mask = np.isnan(Z_ops).any(axis=1)
    n_missing = int(missing_mask.sum())
    if n_missing > 0:
        preview_cols = ["sample_id"]
        if "window_idx" in merged.columns:
            preview_cols.append("window_idx")

        missing_preview = merged.loc[missing_mask, preview_cols].head(10)
        raise ValueError(
            f"NaN values found in operator features after merge. "
            f"Missing rows: {n_missing}/{len(merged)}. "
            f"Example missing keys:\n{missing_preview.to_string(index=False)}"
        )

    if np.isnan(Z_ops).any():
        raise ValueError("NaN values found in operator features after merge.")

    logger.info("Loaded operator features with shape: %s", Z_ops.shape)
    return Z_ops


# =============================================================================
# Hybrid dataset / model / train-eval
# =============================================================================

class SequenceWithOperatorDataset(Dataset):
    def __init__(self, X, y, Z_ops):
        self.X = X
        self.y = y
        self.Z_ops = Z_ops

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)

        y_val = self.y[idx]
        if np.issubdtype(np.asarray(self.y).dtype, np.integer):
            y = torch.tensor(y_val, dtype=torch.long)
        else:
            y = torch.tensor(y_val, dtype=torch.float32)

        z = torch.tensor(self.Z_ops[idx], dtype=torch.float32)
        return x, z, y


class KOLGRURegressor(nn.Module):
    def __init__(self, n_features: int, operator_dim: int, hidden_size: int = 128):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.op_mlp = nn.Sequential(
            nn.Linear(operator_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x_seq, x_ops):
        _, h = self.gru(x_seq)
        seq_emb = h[-1]
        op_emb = self.op_mlp(x_ops)
        fused = torch.cat([seq_emb, op_emb], dim=1)
        return self.head(fused).squeeze(-1)
    

def select_one_window_per_sample_for_kol(
    df: pd.DataFrame,
    X_used: np.ndarray,
    window_s: float,
    f_nom: float = 50.0,
) -> tuple[pd.DataFrame, np.ndarray]:
    if "sample_id" not in df.columns:
        raise ValueError("df must contain 'sample_id'")
    if "dt_start" not in df.columns:
        raise ValueError("df must contain 'dt_start'")
    if "status" not in df.columns:
        raise ValueError("df must contain 'status'")

    work = df.copy().reset_index(drop=True)
    work["_row_idx"] = np.arange(len(work))

    fs = X_used.shape[1] / float(window_s)
    spc = int(np.rint(fs / f_nom))
    T = X_used.shape[1]

    work = work.loc[
        work["status"].astype(str).str.lower() == "fault_start"
    ].copy()

    work["_onset_idx"] = np.rint((-work["dt_start"].astype(float)) * fs).astype(int)

    work["_valid_timing"] = (
        (work["_onset_idx"] >= spc) &
        (work["_onset_idx"] + spc <= T)
    )
    work = work.loc[work["_valid_timing"]].copy()

    target_idx = T // 2
    work["_timing_score"] = np.abs(work["_onset_idx"] - target_idx)

    work = work.sort_values(
        ["sample_id", "_timing_score", "window_idx"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    selected = work.groupby("sample_id", as_index=False).first()

    row_idx = selected["_row_idx"].to_numpy(dtype=int)
    X_sel = X_used[row_idx]

    selected = selected.drop(
        columns=["_row_idx", "_onset_idx", "_valid_timing", "_timing_score"],
        errors="ignore",
    )

    logger.info(
        "KOL subset selection: kept %d rows from %d (unique sample_id=%d)",
        len(selected),
        len(df),
        selected["sample_id"].nunique(),
    )

    return selected.reset_index(drop=True), X_sel


def make_kol_loaders(
    X_used,
    y_all,
    Z_ops,
    idx_train,
    idx_val,
    idx_test,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
):
    ds_train = SequenceWithOperatorDataset(X_used[idx_train], y_all[idx_train], Z_ops[idx_train])
    ds_val = SequenceWithOperatorDataset(X_used[idx_val], y_all[idx_val], Z_ops[idx_val])
    ds_test = SequenceWithOperatorDataset(X_used[idx_test], y_all[idx_test], Z_ops[idx_test])

    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def train_kol_regressor(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs: int = 20,
    patience: int = 15,
):
    criterion = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for x_seq, x_ops, y in train_loader:
            x_seq = x_seq.to(device)
            x_ops = x_ops.to(device)
            y = y.to(device).float()

            optimizer.zero_grad()
            pred = model(x_seq, x_ops)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_seq, x_ops, y in val_loader:
                x_seq = x_seq.to(device)
                x_ops = x_ops.to(device)
                y = y.to(device).float()
                pred = model(x_seq, x_ops)
                val_losses.append(criterion(pred, y).item())

        mean_train = float(np.mean(train_losses)) if train_losses else float("nan")
        mean_val = float(np.mean(val_losses)) if val_losses else float("inf")
        logger.info(
            "epoch %d | train_loss=%.6f | val_loss=%.6f",
            epoch + 1,
            mean_train,
            mean_val,
        )

        if mean_val < best_val:
            best_val = mean_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    if best_state is not None:
        model.load_state_dict(best_state)


def evaluate_kol_regressor(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for x_seq, x_ops, y in test_loader:
            x_seq = x_seq.to(device)
            x_ops = x_ops.to(device)
            pred = model(x_seq, x_ops).cpu().numpy()

            y_pred.append(pred)
            y_true.append(y.numpy())

    y_true = np.concatenate(y_true).astype(np.float64)
    y_pred = np.concatenate(y_pred).astype(np.float64)

    mse = float(np.mean((y_pred - y_true) ** 2))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(mse))

    return {"loss": mse, "mae": mae, "rmse": rmse}, y_true, y_pred


def predict_on_kol_loader(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for x_seq, x_ops, y in loader:
            x_seq = x_seq.to(device)
            x_ops = x_ops.to(device)
            pred = model(x_seq, x_ops).cpu().numpy()

            y_pred.append(pred)
            y_true.append(y.numpy())

    y_true = np.concatenate(y_true).astype(np.float64)
    y_pred = np.concatenate(y_pred).astype(np.float64)
    return y_true, y_pred, None


# =============================================================================
# Main runner
# =============================================================================

@hydra.main(
    version_base=None,
    config_path="../third_party/dl_fault_repo/config",
    config_name="main-config.yaml",
)
def main(config: MainConfig) -> None:
    set_torch_perf_flags()

    X, labels_df, meta = load_windowed_dataset(config)
    device = get_device()
    logger.info("Device: %s", device)

    include_groups = config.training.feature_groups_include
    materialize = config.training.materialize_feature_filters
    X_used, feature_indices_for_ds = maybe_filter_features(
        X=X,
        meta=meta,
        include_groups=include_groups,
        materialize=materialize,
    )

    target_label = str(config.training.target_label)
    if target_label not in L.ALL_TARGETS:
        raise ValueError(f"Unknown target_label='{target_label}'")

    spec = get_task_spec(target_label)
    task_type = infer_task_type_from_spec(spec)
    criterion = spec.criterion
    primary_name = spec.primary_metric
    higher_is_better = spec.higher_is_better

    if config.dataset.topology == "hv_double_line_90kv":
        valid_row_idx = build_valid_row_indices_hv_double_line_90kv(
            labels_df, target_label
        )
        if valid_row_idx is None:
            labels_df_used = labels_df.reset_index(drop=True)
            logger.info(
                "No custom filtering applied for hv_double_line_90kv (valid_row_idx=None)."
            )
        else:
            labels_df_used = labels_df.iloc[valid_row_idx].reset_index(drop=True)
            logger.info(
                "Applied custom filtering for hv_double_line_90kv: kept %d/%d rows (%.2f%%).",
                len(labels_df_used),
                len(labels_df),
                100.0 * (len(labels_df_used) / max(1, len(labels_df))),
            )
    else:
        valid_row_idx = build_valid_row_indices(labels_df, target_label=target_label)
        if valid_row_idx is None:
            labels_df_used = labels_df.reset_index(drop=True)
            logger.info("No target-specific filtering applied (valid_row_idx=None).")
        else:
            labels_df_used = labels_df.iloc[valid_row_idx].reset_index(drop=True)
            logger.info(
                "Applied target-specific filtering: kept %d/%d rows (%.2f%%).",
                len(labels_df_used),
                len(labels_df),
                100.0 * (len(labels_df_used) / max(1, len(labels_df))),
            )
    use_ops = bool(getattr(config.training, "use_operator_features", False))
    if use_ops:
        labels_df_used, X_used = select_one_window_per_sample_for_kol(
            df=labels_df_used,
            X_used=X_used,
            window_s=float(config.window_extraction.window_length),
            f_nom=50.0,
        )

    groups_used = labels_df_used[L.SAMPLE_ID]

    Z_ops = load_operator_features_if_enabled(
        config=config,
        labels_df_used=labels_df_used,
    )

    y_all, class_to_idx = extract_target(labels_df_used, target_label=target_label)

    if task_type == "multiclass":
        y_all, class_to_idx = canonicalize_multiclass_encoding(y_all, task_type)
        logger.info("After canonicalization: %d classes in mapping", len(class_to_idx))

    if np.issubdtype(y_all.dtype, np.floating):
        num_nans = int(np.isnan(y_all).sum())
        if num_nans > 0:
            logger.warning(
                "Target '%s' contains %d NaN values out of %d samples (%.2f%%).",
                target_label,
                num_nans,
                len(y_all),
                (num_nans / len(y_all)) * 100.0,
            )
            raise ValueError("NaN values found in target labels.")

    out_dim = len(class_to_idx) if task_type == "multiclass" else 1

    T, F_eff, flat_dim = infer_input_dims(X_used, feature_indices_for_ds)
    model_name = str(config.model.model_name)
    if Z_ops is not None:
        model_name = f"kol_{model_name}"

    window_s = float(config.window_extraction.window_length)
    window_ms = int(round(1000.0 * window_s))
    step_s = float(getattr(config.window_extraction, "step_length_seconds", np.nan))

    logger.info(
        "Setup: target=%s | task_type=%s | n=%d | T=%d F_eff=%d out_dim=%d | window=%dms (%.3fs) | step=%.3fs | spec=%s",
        target_label,
        task_type,
        len(y_all),
        T,
        F_eff,
        out_dim,
        window_ms,
        window_s,
        step_s,
        spec.log_msg,
    )

    seeds = list(map(int, config.training.seeds))
    if len(seeds) == 0:
        raise RuntimeError("No training seeds provided.")
    seed = int(seeds[0])
    if len(seeds) > 1:
        logger.info("Multiple seeds configured; using first only for CV: %d", seed)

    env_info = get_env_info()

    wb_run = setup_wandb_logging(
        config,
        target_label,
        task_type,
        labels_df_used,
        groups_used,
        F_eff,
        model_name,
        window_s,
        window_ms,
        seed,
        env_info,
    )

    top_out_dir = str(getattr(config.training, "out_dir", "outputs"))
    run_id = wb_run.id if wb_run is not None else env_info["timestamp"]
    run_out_dir = os.path.join(top_out_dir, f"run_{run_id}")
    os.makedirs(run_out_dir, exist_ok=True)

    n_splits = int(getattr(config.training, "n_splits", 5))

    groups_np = (
        groups_used.to_numpy()
        if hasattr(groups_used, "to_numpy")
        else np.asarray(groups_used)
    )

    splits = build_cv_splits_stratified(
        y_all=y_all,
        groups_np=groups_np,
        task_type=task_type,
        n_splits=n_splits,
        seed=seed,
    )

    best_lr, best_wd, eval_only, resave_eval_only = select_best_lr_wd(
        config=config,
        X_used=X_used,
        feature_indices_for_ds=(
            list(feature_indices_for_ds) if feature_indices_for_ds is not None else None
        ),
        task_type=task_type,
        criterion=criterion,
        primary_name=primary_name,
        higher_is_better=higher_is_better,
        valid_row_idx=valid_row_idx,
        labels_df_used=labels_df_used,
        y_all=y_all,
        out_dim=out_dim,
        F_eff=F_eff,
        flat_dim=flat_dim,
        seed=seed,
        n_splits=n_splits,
    )

    if wb_run is not None:
        wb_run.summary["protocol/n_splits"] = int(n_splits)
        wb_run.config.update(
            {"lr_used": float(best_lr), "weight_decay_used": float(best_wd)},
            allow_val_change=True,
        )
        wb_run.summary["opt/lr_used"] = float(best_lr)
        wb_run.summary["opt/wd_used"] = float(best_wd)

    all_fold_metrics: list[dict] = []
    ckpt_dir = str(getattr(config.training, "ckpt_dir", "outputs/checkpoints"))

    for fold_idx, (train_pool_idx, test_idx) in enumerate(splits, start=0):
        train_pool_idx = np.asarray(train_pool_idx, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)

        set_seed(int(seed))

        idx_train, idx_val = split_train_val_from_train_pool(
            groups_used=groups_used,
            train_pool_idx=train_pool_idx,
            val_size=float(config.training.val_size),
            split_seed=int(config.training.split_seed) + int(fold_idx),
        )

        logger.info(
            "[fold %d/%d] split sizes: train=%d | val=%d | test=%d",
            fold_idx + 1,
            n_splits,
            len(idx_train),
            len(idx_val),
            len(test_idx),
        )

        logger.info(
            "=== fold %d/%d | seed %d ===",
            fold_idx + 1,
            n_splits,
            int(seed),
        )

        ckpt_path = os.path.join(
            ckpt_dir,
            f"{config.dataset.topology}__{target_label}__{model_name}__W{window_ms}ms__fold{fold_idx}__seed{seed}.pt",
        )
        os.makedirs(ckpt_dir, exist_ok=True)

        if Z_ops is None:
            train_loader, val_loader, test_loader = make_loaders(
                X_used=X_used,
                y_all=y_all,
                task_type=task_type,
                feature_indices_for_ds=feature_indices_for_ds,
                idx_train=idx_train,
                idx_val=idx_val,
                idx_test=test_idx,
                batch_size=int(config.training.batch_size),
                device=device,
                num_workers=int(config.training.num_workers),
                pin_memory=bool(config.training.pin_memory),
                prefetch_factor=int(config.training.prefetch_factor),
                row_indices=valid_row_idx,
            )

            model = create_model_from_name(
                config,
                n_features=int(F_eff),
                flattened_dim=int(flat_dim),
                out_dim=int(out_dim),
            ).to(device)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(best_lr),
                weight_decay=float(best_wd),
            )

            if eval_only:
                if not os.path.exists(ckpt_path):
                    raise FileNotFoundError(
                        f"eval_only=true but checkpoint not found: {ckpt_path}"
                    )
                ckpt = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(ckpt["model_state_dict"])
                validate_checkpoint_metadata(
                    ckpt=ckpt,
                    expected_topology=str(config.dataset.topology),
                    expected_target=target_label,
                    expected_model=model_name.replace("kol_", ""),
                    expected_window_s=float(config.window_extraction.window_length),
                )
                logger.info("Loaded checkpoint: %s", ckpt_path)
            else:
                train_best_on_val(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                    task_type=task_type,
                    epochs=int(config.training.epochs),
                    primary_name=primary_name,
                    higher_is_better=higher_is_better,
                    binary_threshold=float(config.training.binary_threshold),
                    patience=int(getattr(config.training, "patience", 15)),
                )

            test_metrics = evaluate(
                model,
                test_loader,
                device,
                task_type,
                binary_threshold=float(config.training.binary_threshold),
            )

            y_true_np, y_pred_np, y_score_np = predict_on_loader(
                model=model,
                loader=test_loader,
                device=device,
                task_type=task_type,
                binary_threshold=float(config.training.binary_threshold),
            )

        else:
            if task_type != "regression":
                raise NotImplementedError(
                    "First KOL hybrid version currently supports regression only."
                )

            train_loader, val_loader, test_loader = make_kol_loaders(
                X_used=X_used,
                y_all=y_all,
                Z_ops=Z_ops,
                idx_train=idx_train,
                idx_val=idx_val,
                idx_test=test_idx,
                batch_size=int(config.training.batch_size),
                num_workers=int(config.training.num_workers),
                pin_memory=bool(config.training.pin_memory),
            )

            model = KOLGRURegressor(
                n_features=int(F_eff),
                operator_dim=int(Z_ops.shape[1]),
                hidden_size=int(getattr(config.model, "hidden_size", 128)),
            ).to(device)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(best_lr),
                weight_decay=float(best_wd),
            )

            if eval_only:
                if not os.path.exists(ckpt_path):
                    raise FileNotFoundError(
                        f"eval_only=true but checkpoint not found: {ckpt_path}"
                    )
                ckpt = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(ckpt["model_state_dict"])
                validate_checkpoint_metadata(
                    ckpt=ckpt,
                    expected_topology=str(config.dataset.topology),
                    expected_target=target_label,
                    expected_model=model_name,
                    expected_window_s=float(config.window_extraction.window_length),
                )
                logger.info("Loaded checkpoint: %s", ckpt_path)
            else:
                train_kol_regressor(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    optimizer=optimizer,
                    device=device,
                    epochs=int(config.training.epochs),
                    patience=int(getattr(config.training, "patience", 15)),
                )

            test_metrics, y_true_np, y_pred_np = evaluate_kol_regressor(
                model=model,
                test_loader=test_loader,
                device=device,
            )
            y_score_np = None

        if not eval_only or resave_eval_only:
            save_checkpoint(
                path=ckpt_path,
                model=model,
                config=config,
                meta={
                    "topology": str(config.dataset.topology),
                    "target_label": target_label,
                    "model_name": model_name,
                    "window_length_s": float(config.window_extraction.window_length),
                    "fold_idx": int(fold_idx),
                    "seed": int(seed),
                },
                task_type=task_type,
                target_label=target_label,
                include_groups=list(include_groups) if include_groups is not None else [],
                feature_indices_for_ds=feature_indices_for_ds,
                class_to_idx=class_to_idx if task_type == "multiclass" else None,
                test_metrics=test_metrics,
                X_used_shape=tuple(X_used.shape),
                fold_idx=int(fold_idx),
                seed=int(seed),
            )
            logger.info("Saved checkpoint: %s", ckpt_path)

        logger.info(
            "[fold %d/%d seed %d] test_metrics=%s",
            fold_idx + 1,
            n_splits,
            int(seed),
            test_metrics,
        )

        metrics_row = {
            "fold": int(fold_idx),
            "n_train": int(len(idx_train)),
            "n_val": int(len(idx_val)),
            "n_test": int(len(test_idx)),
            **{f"test/{k}": float(v) for k, v in test_metrics.items()},
        }
        all_fold_metrics.append(metrics_row)

        try:
            split_path = save_fold_splits(
                out_dir=run_out_dir,
                fold_idx=fold_idx,
                idx_train=idx_train,
                idx_val=idx_val,
                idx_test=test_idx,
                labels_df=labels_df_used,
                group_col=L.SAMPLE_ID,
            )
            logger.info("Saved fold splits: %s", split_path)
        except Exception as e:
            logger.warning("Failed to save fold splits: %s", e)

        try:
            meta_cols = [
                L.EVENT_TYPE,
                L.STATUS,
                L.Y_FAULT_LINE,
                L.SAMPLE_ID,
            ]
            pred_path = save_fold_predictions(
                out_dir=run_out_dir,
                fold_idx=fold_idx,
                idx_test=test_idx,
                labels_df=labels_df_used,
                y_true=y_true_np,
                y_pred=y_pred_np,
                y_score=y_score_np,
                task_type=task_type,
                meta_cols=meta_cols,
            )
            logger.info("Saved fold predictions: %s", pred_path)
        except Exception as e:
            logger.warning("Failed to save fold predictions: %s", e)

        try:
            maybe_run_subgroup_analysis(
                task_type=task_type,
                labels_df=labels_df_used,
                valid_row_idx=np.arange(len(labels_df_used)),
                idx_test=test_idx,
                y_true=y_true_np,
                y_pred=y_pred_np,
                out_dir=run_out_dir,
                group_cols=["event_type", "status", "y_fault_line"],
                logger=logger,
                min_support=5,
                regression_add_true_bins=True,
                regression_true_bins_decimals=3,
            )
        except Exception as e:
            logger.warning("Subgroup analysis failed: %s", e)

    metrics_df = pd.DataFrame(all_fold_metrics)
    logger.info("===== CV aggregate metrics =====")
    for col in metrics_df.columns:
        if col.startswith("test/"):
            mean_v = metrics_df[col].mean()
            std_v = metrics_df[col].std(ddof=1) if len(metrics_df) > 1 else 0.0
            logger.info("%s: %.6f ± %.6f", col, mean_v, std_v)

    agg_path = os.path.join(run_out_dir, "cv_metrics_summary.csv")
    metrics_df.to_csv(agg_path, index=False)
    logger.info("Saved aggregate metrics to: %s", agg_path)

    if wb_run is not None:
        for col in metrics_df.columns:
            if col.startswith("test/"):
                wb_run.summary[f"{col}_mean"] = float(metrics_df[col].mean())
                wb_run.summary[f"{col}_std"] = float(
                    metrics_df[col].std(ddof=1) if len(metrics_df) > 1 else 0.0
                )
        wb_run.finish()


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    main()
