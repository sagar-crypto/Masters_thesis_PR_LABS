from __future__ import annotations

import logging
import os

import torch

import hydra
import numpy as np
import pandas as pd
from psp_helper.config import MainConfig
from psp_helper.utils.logging import get_logger
from sklearn.model_selection import GroupKFold

import dl_psp.data.labels as L
from dl_psp.data.data_utils import load_windowed_dataset
from dl_psp.data.features import maybe_filter_features
from dl_psp.data.filters import (
    build_valid_row_indices_hv_double_line_90kv,
    build_valid_row_indices_hv_double_line_110kv,
)
from dl_psp.data.targets import extract_target
from dl_psp.models.model_utils import get_device
from dl_psp.utils.run_utils import (
    get_env_info,
    infer_input_dims,
    save_checkpoint,
    save_fold_splits,
    set_seed,
    set_torch_perf_flags,
)

from KOL.common.cases import derive_fault_case_from_processed_labels, build_case_index
from KOL.common.cv_utils import split_train_val_from_train_pool
from KOL.common.line_utils import attach_line_parameter_metadata
from KOL.common.windowing import (
    select_one_window_per_sample,
    filter_fault_start_windows_only_with_timing,
)
from KOL.datasets.learned_operator_datasets import make_k0_loaders
from KOL.models.learned_operator_models import LearnedOperatorGRU
from KOL.training.learned_operator_train import (
    train_k0_model,
    evaluate_operator_model,
    save_k0_predictions_csv,
)

logger = get_logger(__name__)


def filter_to_single_line(
    labels_df_used: pd.DataFrame,
    X_used: np.ndarray,
    line_filter: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    mask = labels_df_used["y_fault_line"].astype(str) == str(line_filter)

    if int(mask.sum()) == 0:
        raise ValueError(f"No rows found for line_filter='{line_filter}'")

    df_out = labels_df_used.loc[mask].reset_index(drop=True)
    X_out = X_used[mask.to_numpy()]

    logger.info(
        "Filtered to line='%s': kept %d/%d rows (%.2f%%)",
        line_filter,
        len(df_out),
        len(labels_df_used),
        100.0 * len(df_out) / max(1, len(labels_df_used)),
    )
    return df_out, X_out


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
    if target_label != "y_fault_location":
        raise ValueError("This experiment expects training.target_label=y_fault_location")

    topology = str(config.dataset.topology)

    if topology == "hv_double_line_90kv":
        valid_row_idx = build_valid_row_indices_hv_double_line_90kv(
            labels_df, target_label
        )
    elif topology == "hv_double_line_110kv":
        valid_row_idx = build_valid_row_indices_hv_double_line_110kv(
            labels_df, target_label
        )
    else:
        raise ValueError(
            f"This prototype currently supports only hv_double_line_90kv and hv_double_line_110kv, got: {topology}"
        )

    if valid_row_idx is None:
        labels_df_used = labels_df.reset_index(drop=True)
        X_used_filtered = X_used
    else:
        labels_df_used = labels_df.iloc[valid_row_idx].reset_index(drop=True)
        X_used_filtered = X_used[valid_row_idx]

    if topology == "hv_double_line_90kv":
        full_labels_path = str(
            getattr(
                config.training,
                "full_labels_path",
                "/home/vault/iwi5/iwi5305h/new_dataset_90kv/labels.csv",
            )
        )
    elif topology == "hv_double_line_110kv":
        full_labels_path = None
    else:
        raise ValueError(f"No metadata path configured for topology: {topology}")

    labels_df_used = attach_line_parameter_metadata(
        labels_df_used=labels_df_used,
        full_labels_path=full_labels_path,
        topology=topology,
    )

    if topology == "hv_double_line_90kv":
        default_line_filter = "Line_1_2_a"
    elif topology == "hv_double_line_110kv":
        default_line_filter = "MainLn1-2A"
    else:
        raise ValueError(f"No default line filter configured for topology: {topology}")

    line_filter = str(getattr(config.training, "line_filter", default_line_filter))

    labels_df_used = labels_df_used.copy()
    labels_df_used["case"] = labels_df_used.apply(
        derive_fault_case_from_processed_labels,
        axis=1,
    )

    labels_df_used, X_used_filtered = filter_to_single_line(
        labels_df_used=labels_df_used,
        X_used=X_used_filtered,
        line_filter=line_filter,
    )

    window_s = float(config.window_extraction.window_length)
    T_full = X_used_filtered.shape[1]
    fs = T_full / window_s

    window_mode = str(
        getattr(config.training, "k0_window_mode", "all_fault_start")
    ).lower().strip()

    if window_mode == "single_fault_start":
        labels_df_used, X_used_filtered = select_one_window_per_sample(
            df=labels_df_used,
            X_eval=X_used_filtered,
            fs=fs,
            f_nom=50.0,
        )
    elif window_mode == "all_fault_start":
        labels_df_used, X_used_filtered = filter_fault_start_windows_only_with_timing(
            df=labels_df_used,
            X_used=X_used_filtered,
            fs=fs,
            f_nom=50.0,
        )
    else:
        raise ValueError(
            f"Unknown training.k0_window_mode='{window_mode}'. "
            f"Supported: single_fault_start, all_fault_start"
        )

    logger.info("K0 window mode: %s", window_mode)
    logger.info(
        "After K0 window filtering: n=%d | unique sample_id=%d",
        len(labels_df_used),
        int(labels_df_used["sample_id"].nunique()),
    )

    case_idx = build_case_index(labels_df_used)

    y_all, _ = extract_target(labels_df_used, target_label=target_label)
    if np.issubdtype(y_all.dtype, np.floating) and int(np.isnan(y_all).sum()) > 0:
        raise ValueError("NaN values found in target labels.")
    y_all = y_all.astype(np.float32)

    groups_used = labels_df_used[L.SAMPLE_ID]
    groups_np = (
        groups_used.to_numpy()
        if hasattr(groups_used, "to_numpy")
        else np.asarray(groups_used)
    )

    T, F_eff, flat_dim = infer_input_dims(X_used_filtered, feature_indices_for_ds)
    feature_names = meta["feature_names"]

    logger.info(
        "Learned-k0 setup | line=%s | n=%d | T=%d | inferred_fs=%.3f | cases=%s",
        line_filter,
        len(y_all),
        T,
        fs,
        sorted(labels_df_used["case"].unique().tolist()),
    )

    seeds = list(map(int, config.training.seeds))
    if len(seeds) == 0:
        raise RuntimeError("No training seeds provided.")
    seed = int(seeds[0])

    env_info = get_env_info()

    top_out_dir = str(getattr(config.training, "out_dir", "outputs"))
    run_id = env_info["timestamp"]
    safe_line = line_filter.replace("/", "_").replace(" ", "_")
    run_out_dir = os.path.join(top_out_dir, f"{safe_line}__learned_k0__run_{run_id}")
    os.makedirs(run_out_dir, exist_ok=True)

    n_splits = int(getattr(config.training, "n_splits", 5))
    splits = list(
        GroupKFold(n_splits=n_splits).split(
            np.zeros(len(y_all)),
            y_all,
            groups_np,
        )
    )

    all_fold_metrics: list[dict] = []
    ckpt_dir = str(getattr(config.training, "ckpt_dir", "outputs/checkpoints"))
    os.makedirs(ckpt_dir, exist_ok=True)

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

        train_loader, val_loader, test_loader = make_k0_loaders(
            X_used=X_used_filtered,
            labels_df_used=labels_df_used,
            y_all=y_all,
            case_idx=case_idx,
            idx_train=idx_train,
            idx_val=idx_val,
            idx_test=test_idx,
            feature_names=feature_names,
            topology=topology,
            batch_size=int(config.training.batch_size),
            num_workers=int(config.training.num_workers),
            pin_memory=bool(config.training.pin_memory),
        )

        model = LearnedOperatorGRU(
            n_features=6,
            hidden_size=int(getattr(config.model, "hidden_size", 256)),
            num_layers=int(getattr(config.model, "num_layers", 2)),
            dropout=float(getattr(config.model, "dropout", 0.1)),
            bidirectional=bool(getattr(config.model, "bidirectional", False)),
            n_cases=10,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config.training.learning_rate),
            weight_decay=float(config.training.weight_decay),
        )

        ckpt_path = os.path.join(
            ckpt_dir,
            f"{config.dataset.topology}__{safe_line}__learned_k0__fold{fold_idx}__seed{seed}.pt",
        )

        eval_only = bool(getattr(config.training, "eval_only", False))
        resave_eval_only = bool(getattr(config.training, "resave_eval_only", False))

        if eval_only:
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(
                    f"eval_only=true but checkpoint not found: {ckpt_path}"
                )
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            logger.info("Loaded checkpoint: %s", ckpt_path)

            test_metrics, y_true_np, y_pred_np, aux, case_metrics_df = evaluate_operator_model(
                model=model,
                test_loader=test_loader,
                device=device,
                fs=fs,
            )
        else:
            train_k0_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                device=device,
                fs=fs,
                epochs=int(getattr(config.training, "epochs", 50)),
                patience=int(getattr(config.training, "patience", 10)),
                use_scheduler=bool(getattr(config.training, "use_scheduler", True)),
                scheduler_factor=float(getattr(config.training, "scheduler_factor", 0.5)),
                scheduler_patience=int(getattr(config.training, "scheduler_patience", 3)),
                scheduler_min_lr=float(getattr(config.training, "scheduler_min_lr", 1e-6)),
            )

            test_metrics, y_true_np, y_pred_np, aux, case_metrics_df = evaluate_operator_model(
                model=model,
                test_loader=test_loader,
                device=device,
                fs=fs,
            )

        if not eval_only or resave_eval_only:
            save_checkpoint(
                path=ckpt_path,
                model=model,
                config=config,
                meta={
                    "topology": str(config.dataset.topology),
                    "target_label": target_label,
                    "model_name": "learned_k0",
                    "window_length_s": float(config.window_extraction.window_length),
                    "fold_idx": int(fold_idx),
                    "seed": int(seed),
                    "line_filter": line_filter,
                    "ground_cases_only": True,
                },
                task_type="regression",
                target_label=target_label,
                include_groups=list(include_groups) if include_groups is not None else [],
                feature_indices_for_ds=feature_indices_for_ds,
                class_to_idx=None,
                test_metrics=test_metrics,
                X_used_shape=tuple(X_used_filtered.shape),
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

        try:
            case_metrics_path = os.path.join(run_out_dir, f"fold{fold_idx}_case_metrics.csv")
            case_metrics_df.to_csv(case_metrics_path, index=False)
            logger.info("Saved per-case metrics: %s", case_metrics_path)
        except Exception as e:
            logger.warning("Failed to save per-case metrics: %s", e)

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
            pred_path = save_k0_predictions_csv(
                out_dir=run_out_dir,
                fold_idx=fold_idx,
                idx_test=test_idx,
                labels_df=labels_df_used,
                y_true=y_true_np,
                y_pred=y_pred_np,
                aux=aux,
            )
            logger.info("Saved fold predictions: %s", pred_path)
        except Exception as e:
            logger.warning("Failed to save fold predictions: %s", e)

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


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    main()
