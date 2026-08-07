from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Sequence
from typing import Optional

import numpy as np
import pandas as pd
import wandb
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from psp_helper.config import MainConfig

from dl_psp.utils.tuning_utils import tune_lr_wd_on_single_fold


class CohortAuditError(ValueError):
    """Raised when prepared data does not match the canonical protocol."""


@dataclass(frozen=True)
class CohortAudit:
    rows: int
    events: int
    folds: int
    windows: list[int] | None


def audit_cohort(
    labels: pd.DataFrame,
    splits: Iterable[tuple[Sequence[int], Sequence[int]]],
    *,
    expected_rows: int,
    expected_events: int,
    expected_folds: int = 5,
    group_column: str = "sample_id",
    expected_windows: Sequence[int] | None = None,
    prior_values=None,
    prior_column: str | None = None,
    operator_columns: Sequence[str] = (),
) -> CohortAudit:
    if group_column not in labels.columns:
        raise CohortAuditError(f"missing group column: {group_column}")
    rows = len(labels)
    events = int(labels[group_column].nunique(dropna=False))
    if rows != int(expected_rows):
        raise CohortAuditError(f"cohort rows mismatch: expected {expected_rows}, observed {rows}")
    if events != int(expected_events):
        raise CohortAuditError(f"cohort events mismatch: expected {expected_events}, observed {events}")

    required = ([prior_column] if prior_column else []) + list(operator_columns)
    missing = [name for name in required if name not in labels.columns]
    if missing:
        raise CohortAuditError(f"missing operator columns: {missing}")
    if prior_values is not None and not np.isfinite(np.asarray(prior_values, dtype=float)).all():
        raise CohortAuditError("prepared prior contains non-finite values")

    observed_windows = None
    if expected_windows is not None:
        if "window_idx" not in labels.columns:
            raise CohortAuditError("configured window indices but window_idx is missing")
        observed_windows = sorted(map(int, pd.unique(labels["window_idx"])))
        wanted = sorted(map(int, expected_windows))
        if observed_windows != wanted:
            raise CohortAuditError(f"window set mismatch: expected {wanted}, observed {observed_windows}")

    split_list = list(splits)
    if len(split_list) != int(expected_folds):
        raise CohortAuditError(f"outer folds mismatch: expected {expected_folds}, observed {len(split_list)}")
    all_rows = set(range(rows))
    test_counts = np.zeros(rows, dtype=np.int16)
    group_fold_counts = {}
    for fold, (train, test) in enumerate(split_list):
        train_set, test_set = set(map(int, train)), set(map(int, test))
        if not train_set or not test_set or train_set | test_set != all_rows or train_set & test_set:
            raise CohortAuditError(f"fold {fold} is not a complete disjoint outer split")
        if set(labels.iloc[list(train_set)][group_column]) & set(labels.iloc[list(test_set)][group_column]):
            raise CohortAuditError(f"sample_id crosses train/test boundary in fold {fold}")
        test_counts[list(test_set)] += 1
        for group in pd.unique(labels.iloc[np.asarray(test, dtype=int)][group_column]):
            group_fold_counts.setdefault(group, set()).add(fold)
    if not np.all(test_counts == 1):
        raise CohortAuditError("each row/sample_id must occur in exactly one outer test fold")
    if any(len(v) != 1 for v in group_fold_counts.values()) or len(group_fold_counts) != events:
        raise CohortAuditError("each sample_id must belong to exactly one outer test fold")
    return CohortAudit(rows, events, len(split_list), observed_windows)


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
        i for i in train_pool_idx
        if (groups_used.iloc[i] if hasattr(groups_used, "iloc") else groups_used[i]) in val_groups
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
    labels_df: Optional[pd.DataFrame] = None,
    cv_mode: str = "group",
    stratify_col: str = "y_fault_location",
):
    """
    Build grouped CV splits.

    cv_mode options
    ---------------
    group:
        Plain GroupKFold. Keeps groups together, but can accidentally create
        location-held-out folds if sample_id ordering is structured.

    stratified_location:
        StratifiedGroupKFold using labels_df[stratify_col], usually y_fault_location.
        Keeps groups together and balances fault locations across folds.

    stratified_case:
        StratifiedGroupKFold using labels_df["case"].

    stratified_location_case:
        StratifiedGroupKFold using y_fault_location + case.
        This can fail if some location-case combinations have fewer groups than n_splits.

    auto:
        For multiclass, stratifies using y_all.
        For regression, falls back to group unless labels_df and stratify_col are provided.
    """
    groups_np = np.asarray(groups_np)
    cv_mode = str(cv_mode).lower().strip()

    if cv_mode == "auto":
        if task_type == "multiclass":
            cv_mode = "multiclass"
        elif labels_df is not None and stratify_col in labels_df.columns:
            cv_mode = "stratified_location"
        else:
            cv_mode = "group"

    if cv_mode == "group":
        splitter = GroupKFold(n_splits=n_splits)
        return list(splitter.split(np.zeros(len(y_all)), y_all, groups_np))

    if cv_mode == "multiclass":
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        return list(splitter.split(np.zeros(len(y_all)), y_all, groups_np))

    if labels_df is None:
        raise ValueError(
            f"labels_df is required for cv_mode='{cv_mode}'."
        )

    if cv_mode == "stratified_location":
        if stratify_col not in labels_df.columns:
            raise ValueError(
                f"labels_df does not contain stratify_col='{stratify_col}'. "
                f"Available columns: {list(labels_df.columns)}"
            )

        y_strat = labels_df[stratify_col].astype(str).to_numpy()

        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        return list(splitter.split(np.zeros(len(y_all)), y_strat, groups_np))

    if cv_mode == "stratified_case":
        if "case" not in labels_df.columns:
            raise ValueError("labels_df must contain 'case' for cv_mode='stratified_case'.")

        y_strat = labels_df["case"].astype(str).to_numpy()

        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        return list(splitter.split(np.zeros(len(y_all)), y_strat, groups_np))

    if cv_mode == "stratified_location_case":
        if stratify_col not in labels_df.columns:
            raise ValueError(
                f"labels_df does not contain stratify_col='{stratify_col}'."
            )
        if "case" not in labels_df.columns:
            raise ValueError(
                "labels_df must contain 'case' for cv_mode='stratified_location_case'."
            )

        y_strat = (
            labels_df[stratify_col].astype(str)
            + "__"
            + labels_df["case"].astype(str)
        ).to_numpy()

        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        return list(splitter.split(np.zeros(len(y_all)), y_strat, groups_np))

    raise ValueError(
        f"Unknown cv_mode='{cv_mode}'. Supported modes: "
        "group, auto, multiclass, stratified_location, stratified_case, stratified_location_case"
    )


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
    logger,
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
        logger.info("Tuning selected lr=%g wd=%g | info=%s", best_lr, best_wd, tune_info)

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
        n_groups = int(groups_used.nunique()) if isinstance(groups_used, pd.Series) else len(np.unique(groups_used))
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
    logger,
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
        mismatches.append(f"window_length: checkpoint={ckpt_window}, expected={expected_window_s}")

    if mismatches:
        logger.warning("Checkpoint metadata mismatch detected:\n  %s", "\n  ".join(mismatches))
