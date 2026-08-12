"""Explicit bridge from the public Hydra contract to the private PSP schema."""
from __future__ import annotations

import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from .common import REPO_ROOT

PRIVATE_ROOT = REPO_ROOT / "third_party" / "dl_fault_repo"
PRIVATE_CONFIG = PRIVATE_ROOT / "config" / "main-config.yaml"


def _private_base(topology: str) -> DictConfig:
    """Compose the private defaults without invoking its Hydra command wrapper."""
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base="1.3", config_dir=str(PRIVATE_ROOT / "config")):
        return compose(config_name="main-config", overrides=[f"dataset={topology}"])


def to_private_config(cfg: DictConfig, *, output_dir: Path | None = None) -> DictConfig:
    """Map a canonical public experiment onto the private runner schema.

    The adapter chooses the voltage-specific topology, fixes the private model
    input to 48 channels, and translates P/G/C/L family choices into physics,
    waveform-only, or operator-prior execution.  Hybrid priors remain percentage
    values on disk; downstream loading normalizes them for the neural models.
    Canonical cohort dimensions, split controls, and path overrides are copied
    into the legacy tree so its existing runners can enforce the public contract.

    Args:
        cfg: Resolved public Hydra configuration.
        output_dir: Optional run directory replacing ``cfg.paths.output_root``.

    Returns:
        A resolved, non-struct ``DictConfig`` accepted by the private runners.

    Raises:
        ValueError: If required mapped values or unresolved interpolations remain.
    """
    topology = "hv_double_line_90kv" if int(cfg.dataset.voltage_kv) == 90 else "hv_double_line_110kv"
    legacy = _private_base(topology)
    # The private Hydra tree is composed in struct mode; the compatibility
    # surface intentionally adds a small set of opt-in canonical keys.
    OmegaConf.set_struct(legacy, False)
    out = Path(output_dir or str(cfg.paths.output_root)).resolve()
    is_gru = str(cfg.model.family) == "waveform_gru"
    is_physics = str(cfg.model.family) == "physics_only"
    patch = {
        "dataset": {
            "topology": topology,
            "dataset_directory": str(Path(str(cfg.paths.data_root)).resolve()),
        },
        "model": {
            "model_name": "gru_regressor",
            "input_size": 48,
            "hidden_size": int(cfg.model.get("hidden_size", 128) or 128),
            "num_layers": int(cfg.model.get("num_layers", 2) or 2),
            "dropout": float(cfg.model.get("dropout", 0.1) or 0.0),
            "bidirectional": False,
        },
        "training": {
            "target_label": str(cfg.target), "n_splits": int(cfg.folds),
            "val_size": float(cfg.validation_fraction), "split_seed": int(cfg.seed),
            "seeds": [int(cfg.seed)], "batch_size": cfg.training.batch_size,
            "epochs": int(cfg.execution.max_epochs or cfg.training.epochs),
            "patience": cfg.training.patience, "learning_rate": cfg.training.learning_rate,
            "weight_decay": cfg.training.weight_decay, "out_dir": str(out),
            "ckpt_dir": str(out / "checkpoints"), "eval_only": bool(cfg.execution.evaluate_only),
            "use_operator_features": not is_gru and not is_physics,
            "operator_features_path": str(Path(str(cfg.dataset.model_input_path)).resolve()),
            "operator_prior_col": str(cfg.prior.operator_prior_col),
            "operator_feature_cols": list(cfg.prior.operator_features),
            "kol_model_mode": str(cfg.training.get("kol_model_mode", "gru_only" if is_gru else "legacy_residual")),
            "kol_prediction_mode": str(cfg.training.get("kol_prediction_mode", "plain")),
            "kol_window_mode": str(cfg.protocol.cohort_mode),
            "operator_window_mode": str(cfg.protocol.cohort_mode),
            "operator_side_mode": "two_ended_posseq" if str(cfg.prior.mode) == "two_ended" else "default",
            "line_filter": cfg.training.get("line_filter", None),
            "tune_lr_wd": False,
            "feature_groups_include": (
                ["lines"]
                if int(cfg.dataset.voltage_kv) == 110 and not is_physics
                else ["lines", "loads", "winds", "extgrid"]
            ),
            "materialize_feature_filters": False,
            "input_representation": "waveform",
            "cv_mode": "group" if is_gru else "stratified_location",
            "cv_stratify_col": "y_fault_location",
            "case_emb_dim": int(cfg.model.get("case_embedding", 8) or 8),
            "fusion_head_hidden_size": int(cfg.model.get("fusion_head", 64) or 64),
            "bounded_residual_max": float(cfg.model.get("residual_bound", 1.0) or 1.0),
            "gate_init_bias": float(cfg.model.get("gate_bias", -3.0)),
            "canonical_expected_rows": int(cfg.protocol.expected_rows),
            "canonical_expected_events": int(cfg.protocol.expected_events),
            "canonical_expected_timesteps": int(
                cfg.protocol.get("expected_timesteps", 0) or 0
            ),
            "canonical_expected_features": int(
                cfg.protocol.get("expected_features", 0) or 0
            ),
            "canonical_window_indices": cfg.protocol.window_indices,
            "canonical_hard_audit": True,
            "canonical_experiment_id": str(cfg.experiment.id),
            "canonical_fold": cfg.execution.fold,
            "max_train_batches": cfg.execution.max_train_batches,
            "max_val_batches": cfg.execution.max_val_batches,
            "max_test_batches": cfg.execution.max_test_batches,
        },
        "window_extraction": {
            "windows_local_dir": str(Path(str(cfg.dataset.waveform_path)).resolve()),
            "window_length": float(cfg.protocol.window_length_ms) / 1000.0,
            "step_length_seconds": float(cfg.protocol.step_ms) / 1000.0,
        },
        "tracking": {"use_wandb": not bool(cfg.execution.disable_tracking)},
    }
    result = OmegaConf.merge(legacy, patch)
    OmegaConf.resolve(result)
    required = ("dataset.topology", "training.target_label", "training.out_dir", "model.hidden_size")
    missing = [key for key in required if OmegaConf.select(result, key) is None]
    if missing or "${" in OmegaConf.to_yaml(result, resolve=False):
        raise ValueError(f"incomplete canonical adapter; missing={missing}")
    return result


def enable_private_imports() -> None:
    src = str(PRIVATE_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
