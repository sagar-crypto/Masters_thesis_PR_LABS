from __future__ import annotations

import logging
import os

import hydra
import numpy as np
import pandas as pd
import torch
from psp_helper.config import MainConfig
from psp_helper.utils.logging import get_logger

import dl_psp.data.labels as L
from dl_psp.data.data_utils import load_windowed_dataset
from dl_psp.data.features import maybe_filter_features
from dl_psp.data.filters import (
    build_valid_row_indices,
    build_valid_row_indices_hv_double_line_90kv,
    build_valid_row_indices_hv_double_line_110kv,
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

from KOL.common.constants import CASE_TO_IDX
from KOL.common.cases import build_case_index
from KOL.common.cv_utils import (
    build_cv_splits_stratified,
    canonicalize_multiclass_encoding,
    select_best_lr_wd,
    setup_wandb_logging,
    split_train_val_from_train_pool,
    validate_checkpoint_metadata,
)
from KOL.common.windowing import (
    filter_fault_start_windows_only,
    filter_to_single_line_if_enabled,
    get_kol_mode,
    select_one_window_per_sample_for_kol,
    filter_fault_start_windows_only_with_timing
)
from KOL.common.operator_features import load_operator_inputs_if_enabled
from KOL.datasets.kol_datasets import make_kol_loaders
from KOL.models.kol_residual_models import KOLGRUCaseResidualRegressor
from KOL.training.kol_residual_train import (
    evaluate_kol_case_k0,
    predict_on_kol_case_k0,
    train_kol_case_k0,
)

logger = get_logger(__name__)
logger.info(
    "DEBUG save_fold_predictions loaded from: %s",
    save_fold_predictions.__code__.co_filename,
)


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

    topology = str(config.dataset.topology)

    if topology == "hv_double_line_90kv":
        valid_row_idx = build_valid_row_indices_hv_double_line_90kv(
            labels_df, target_label
        )
        if valid_row_idx is None:
            labels_df_used = labels_df.reset_index(drop=True)
            X_used_filtered = X_used
            logger.info(
                "No custom filtering applied for hv_double_line_90kv (valid_row_idx=None)."
            )
        else:
            labels_df_used = labels_df.iloc[valid_row_idx].reset_index(drop=True)
            X_used_filtered = X_used[valid_row_idx]
            logger.info(
                "Applied custom filtering for hv_double_line_90kv: kept %d/%d rows (%.2f%%).",
                len(labels_df_used),
                len(labels_df),
                100.0 * (len(labels_df_used) / max(1, len(labels_df))),
            )

    elif topology == "hv_double_line_110kv":
        valid_row_idx = build_valid_row_indices_hv_double_line_110kv(
            labels_df, target_label
        )
        if valid_row_idx is None:
            labels_df_used = labels_df.reset_index(drop=True)
            X_used_filtered = X_used
            logger.info(
                "No custom filtering applied for hv_double_line_110kv (valid_row_idx=None)."
            )
        else:
            labels_df_used = labels_df.iloc[valid_row_idx].reset_index(drop=True)
            X_used_filtered = X_used[valid_row_idx]
            logger.info(
                "Applied custom filtering for hv_double_line_110kv: kept %d/%d rows (%.2f%%).",
                len(labels_df_used),
                len(labels_df),
                100.0 * (len(labels_df_used) / max(1, len(labels_df))),
            )

    else:
        valid_row_idx = build_valid_row_indices(labels_df, target_label=target_label)
        if valid_row_idx is None:
            labels_df_used = labels_df.reset_index(drop=True)
            X_used_filtered = X_used
            logger.info("No target-specific filtering applied (valid_row_idx=None).")
        else:
            labels_df_used = labels_df.iloc[valid_row_idx].reset_index(drop=True)
            X_used_filtered = X_used[valid_row_idx]
            logger.info(
                "Applied target-specific filtering: kept %d/%d rows (%.2f%%).",
                len(labels_df_used),
                len(labels_df),
                100.0 * (len(labels_df_used) / max(1, len(labels_df))),
            )

    labels_df_used, X_used_filtered = filter_to_single_line_if_enabled(
        labels_df_used=labels_df_used,
        X_used=X_used_filtered,
        config=config,
        logger=logger,
    )

    use_ops = bool(getattr(config.training, "use_operator_features", False))
    kol_window_mode = str(getattr(config.training, "kol_window_mode", "single_fault_start"))

    if use_ops:
        if kol_window_mode == "single_fault_start":
            labels_df_used, X_used_filtered = select_one_window_per_sample_for_kol(
                df=labels_df_used,
                X_used=X_used_filtered,
                window_s=float(config.window_extraction.window_length),
                f_nom=50.0
            )
        elif kol_window_mode == "all_fault_start":
            window_s = float(config.window_extraction.window_length)
            fs = X_used_filtered.shape[1] / window_s

            labels_df_used, X_used_filtered = filter_fault_start_windows_only_with_timing(
                df=labels_df_used,
                X_used=X_used_filtered,
                fs=fs,
                f_nom=50.0,
            )

            logger.info(
                "KOL window mode = all_fault_start with timing filter: kept %d rows | unique sample_id=%d",
                len(labels_df_used),
                int(labels_df_used[L.SAMPLE_ID].nunique()),
            )

            if "window_idx" in labels_df_used.columns:
                logger.info(
                    "Window_idx distribution after timing filter:\n%s",
                    labels_df_used["window_idx"].value_counts().sort_index().to_string(),
                )
        elif kol_window_mode == "all_valid":
            logger.info("KOL window mode = all_valid (no extra window filtering applied).")
        else:
            raise ValueError(
                f"Unknown training.kol_window_mode='{kol_window_mode}'. "
                f"Supported: single_fault_start, all_fault_start, all_valid"
            )

    groups_used = labels_df_used[L.SAMPLE_ID]

    d_phys_prior, op_features, operator_feature_cols = load_operator_inputs_if_enabled(
        config=config,
        labels_df_used=labels_df_used
    )
    case_idx = build_case_index(labels_df_used)
    idx_to_case = {v: k for k, v in CASE_TO_IDX.items()}
    labels_df_used = labels_df_used.copy()
    labels_df_used["case"] = [idx_to_case[int(i)] for i in case_idx]
    kol_prediction_mode = get_kol_mode(config)

    logger.info("KOL prediction mode: %s", kol_prediction_mode)
    logger.info("Selected operator feature columns: %s", operator_feature_cols)
    logger.info("KOL window mode: %s", kol_window_mode if use_ops else "n/a")

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

    if task_type == "multiclass":
        if class_to_idx is None:
            raise ValueError("class_to_idx is None for multiclass task.")
        out_dim = len(class_to_idx)
    else:
        out_dim = 1

    T, F_eff, flat_dim = infer_input_dims(X_used_filtered, feature_indices_for_ds)
    model_name = str(config.model.model_name)
    kol_mode = get_kol_mode(config) if d_phys_prior is not None else None
    if d_phys_prior is not None:
        model_name = f"kol_{kol_mode}_{model_name}"

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

    if topology == "hv_double_line_90kv":
        default_line_filter = "Line_1_2_a"
    elif topology == "hv_double_line_110kv":
        default_line_filter = "MainLn1-2A"
    else:
        default_line_filter = "all_lines"

    line_filter = str(getattr(config.training, "line_filter", default_line_filter))
    kol_mode_name = str(getattr(config.training, "kol_prediction_mode", "plain"))
    window_mode_name = str(getattr(config.training, "kol_window_mode", "default"))
    operator_path = str(getattr(config.training, "operator_features_path", "no_operator_file"))

    operator_tag = os.path.splitext(os.path.basename(operator_path))[0]
    operator_tag = operator_tag.replace("/", "_").replace(" ", "_")

    safe_line = line_filter.replace("/", "_").replace(" ", "_")
    safe_kol = kol_mode_name.replace("/", "_").replace(" ", "_")
    safe_window = window_mode_name.replace("/", "_").replace(" ", "_")

    run_out_dir = os.path.join(
        top_out_dir,
        f"{safe_line}__{operator_tag}__{safe_kol}__{safe_window}__run_{run_id}",
    )
    os.makedirs(run_out_dir, exist_ok=True)

    n_splits = int(getattr(config.training, "n_splits", 5))

    groups_np = (
        groups_used.to_numpy()
        if hasattr(groups_used, "to_numpy")
        else np.asarray(groups_used)
    )

    cv_mode = str(getattr(config.training, "cv_mode", "group")).lower().strip()
    cv_stratify_col = str(getattr(config.training, "cv_stratify_col", "y_fault_location"))

    logger.info(
        "CV setup | mode=%s | stratify_col=%s | n_splits=%d",
        cv_mode,
        cv_stratify_col,
        n_splits,
    )

    splits = build_cv_splits_stratified(
        y_all=y_all,
        groups_np=groups_np,
        task_type=task_type,
        n_splits=n_splits,
        seed=int(config.training.split_seed),
        labels_df=labels_df_used,
        cv_mode=cv_mode,
        stratify_col=cv_stratify_col,
    )

    best_lr, best_wd, eval_only, resave_eval_only = select_best_lr_wd(
        config=config,
        X_used=X_used_filtered,
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
        logger=logger,
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

        if d_phys_prior is None:
            train_loader, val_loader, test_loader = make_loaders(
                X_used=X_used_filtered,
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
                    logger=logger,
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
                    "First KOL residual version currently supports regression only."
                )

            train_loader, val_loader, test_loader = make_kol_loaders(
                X_used=X_used_filtered,
                y_all=y_all,
                d_phys_prior=d_phys_prior,
                case_idx=case_idx,
                op_features=op_features,
                idx_train=idx_train,
                idx_val=idx_val,
                idx_test=test_idx,
                feature_indices_for_ds=feature_indices_for_ds,
                batch_size=int(config.training.batch_size),
                num_workers=int(config.training.num_workers),
                pin_memory=bool(config.training.pin_memory),
            )

            logger.info("op_features is None: %s", op_features is None)
            if op_features is not None:
                logger.info("op_features shape: %s", op_features.shape)
                logger.info("n_op_features passed to model: %d", int(op_features.shape[1]))

            model = KOLGRUCaseResidualRegressor(
                n_features=int(F_eff),
                n_op_features=0 if op_features is None else int(op_features.shape[1]),
                hidden_size=int(getattr(config.model, "hidden_size", 128)),
                num_layers=int(getattr(config.model, "num_layers", 2)),
                dropout=float(getattr(config.model, "dropout", 0.1)),
                bidirectional=bool(getattr(config.model, "bidirectional", False)),
                n_cases=len(CASE_TO_IDX),
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
                    logger=logger,
                )
                logger.info("Loaded checkpoint: %s", ckpt_path)
            else:
                if kol_prediction_mode == "prior_only":
                    logger.info("KOL mode is prior_only: skipping training.")
                else:
                    train_kol_case_k0(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        optimizer=optimizer,
                        device=device,
                        prediction_mode=kol_prediction_mode,
                        epochs=int(config.training.epochs),
                        patience=int(getattr(config.training, "patience", 15)),
                        logger=logger,
                    )

            test_metrics, y_true_np, y_pred_np, residual_np, dprior_np, case_np = evaluate_kol_case_k0(
                model=model,
                test_loader=test_loader,
                device=device,
                prediction_mode=kol_prediction_mode,
                logger=logger,
            )

            y_score_np = residual_np

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
            if "window_idx" in labels_df_used.columns:
                meta_cols.append("window_idx")

            if "case" in labels_df_used.columns:
                meta_cols.append("case")

            logger.info(
                "DEBUG about to save predictions | fold=%d | run_out_dir=%s | len(test_idx)=%d | y_true_shape=%s | y_pred_shape=%s | y_score_shape=%s",
                fold_idx,
                run_out_dir,
                len(test_idx),
                np.asarray(y_true_np).shape,
                np.asarray(y_pred_np).shape,
                np.asarray(y_score_np).shape,
            )
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
                extra_cols={
                    "d_prior": dprior_np,
                    "residual": residual_np,
                    "case_idx": case_np,
                } if d_phys_prior is not None else None,
            )
            logger.info("Saved fold predictions: %s", pred_path)
        except Exception as e:
            logger.warning("Failed to save fold predictions: %s", e)

        try:
            fold_analysis_dir = os.path.join(run_out_dir, f"fold{fold_idx}_subgroups")
            os.makedirs(fold_analysis_dir, exist_ok=True)

            maybe_run_subgroup_analysis(
                task_type=task_type,
                labels_df=labels_df_used,
                valid_row_idx=np.arange(len(labels_df_used)),
                idx_test=test_idx,
                y_true=y_true_np,
                y_pred=y_pred_np,
                out_dir=fold_analysis_dir,
                group_cols=["event_type", "status", "y_fault_line", "case"],
                logger=logger,
                min_support=5,
                regression_add_true_bins=True,
                regression_true_bins_decimals=3,
            )

            logger.info(
                "Saved subgroup analysis for fold %d to: %s",
                fold_idx,
                fold_analysis_dir,
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
