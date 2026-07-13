from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import wandb
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from psp_helper.config import MainConfig

from dl_psp.utils.tuning_utils import tune_lr_wd_on_single_fold


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
        y_strat = _make_valid_location_strata(
            labels_df=labels_df,
            groups_np=groups_np,
            stratify_col=stratify_col,
            n_splits=n_splits,
        )

        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )

        return list(
            splitter.split(
                np.zeros(len(y_all)),
                y_strat,
                groups_np,
            )
        )
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

def _make_valid_location_strata(
    *,
    labels_df: pd.DataFrame,
    groups_np: np.ndarray,
    stratify_col: str,
    n_splits: int,
    max_bins: int = 10,
) -> np.ndarray:
    """
    Create valid labels for StratifiedGroupKFold.

    First try exact fault-location values. If an exact location has fewer
    than n_splits event groups, fall back to deterministic quantile bins
    of the event-level fault locations.
    """
    if stratify_col not in labels_df.columns:
        raise ValueError(
            f"labels_df does not contain stratify_col='{stratify_col}'. "
            f"Available columns: {list(labels_df.columns)}"
        )

    if len(labels_df) != len(groups_np):
        raise ValueError(
            "labels_df and groups_np must have the same length."
        )

    work = pd.DataFrame(
        {
            "group": np.asarray(groups_np),
            "location": pd.to_numeric(
                labels_df[stratify_col],
                errors="coerce",
            ),
        }
    )

    if not np.isfinite(work["location"].to_numpy()).all():
        raise ValueError(
            f"Column '{stratify_col}' contains non-finite values."
        )

    # Every event/sample_id must correspond to one true fault location.
    locations_per_group = (
        work.groupby("group", sort=False)["location"]
        .nunique(dropna=False)
    )

    invalid_groups = locations_per_group[
        locations_per_group > 1
    ]

    if not invalid_groups.empty:
        raise ValueError(
            f"Some event groups have multiple '{stratify_col}' values. "
            f"Examples: {invalid_groups.head(10).to_dict()}"
        )

    # One location per event/sample_id.
    group_locations = (
        work.drop_duplicates(subset="group", keep="first")
        .set_index("group")["location"]
    )

    # First preserve the original exact-location behavior whenever possible.
    exact_labels = group_locations.astype(str)
    exact_counts = exact_labels.value_counts()
    min_exact_count = int(exact_counts.min())

    if min_exact_count >= int(n_splits):
        group_to_stratum = exact_labels

        y_strat = pd.Series(groups_np).map(group_to_stratum)

        if y_strat.isna().any():
            raise RuntimeError(
                "Could not map exact location strata back to all rows."
            )

        print(
            f"CV stratification: using exact '{stratify_col}' values. "
            f"Minimum groups per location: {min_exact_count}."
        )

        return y_strat.astype(str).to_numpy()

    # Exact locations are too sparse. Create deterministic quantile bins.
    n_groups = int(len(group_locations))

    max_possible_bins = min(
        int(max_bins),
        int(n_groups // int(n_splits)),
    )

    for n_bins in range(max_possible_bins, 1, -1):
        try:
            group_bins = pd.qcut(
                group_locations,
                q=n_bins,
                labels=False,
                duplicates="drop",
            )
        except ValueError:
            continue

        group_bins = pd.Series(
            group_bins,
            index=group_locations.index,
        )

        if group_bins.isna().any():
            continue

        bin_counts = group_bins.value_counts()

        if (
            int(group_bins.nunique()) >= 2
            and int(bin_counts.min()) >= int(n_splits)
        ):
            group_to_stratum = group_bins.astype(int).astype(str)

            y_strat = pd.Series(groups_np).map(group_to_stratum)

            if y_strat.isna().any():
                raise RuntimeError(
                    "Could not map location-bin strata back to all rows."
                )

            print(
                f"CV stratification: exact '{stratify_col}' values were "
                f"too sparse for n_splits={n_splits} "
                f"(minimum groups per exact location: {min_exact_count}). "
                f"Using {int(group_bins.nunique())} quantile location bins."
            )

            return y_strat.astype(str).to_numpy()

    raise ValueError(
        f"Could not create valid location strata for '{stratify_col}' "
        f"with n_splits={n_splits}. "
        f"Number of event groups: {n_groups}. "
        f"Minimum groups per exact location: {min_exact_count}."
    )
