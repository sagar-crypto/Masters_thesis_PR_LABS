from __future__ import annotations

import numpy as np
from typing import Any
from psp_helper.config import MainConfig



def build_sliding_phasor_sequence(
    X: np.ndarray,
    feature_names: list[str],
    fs: float,
    f_nom: float = 50.0,
    step_samples: int = 16,
    mode: str = "real_imag",
) -> tuple[np.ndarray, list[str]]:
    """
    Convert raw waveform windows into a sliding one-cycle phasor sequence.

    Input:
        X: shape (N, T, C)

    Output:
        X_ph: shape (N, S, C_new)

    Each phasor time step is computed from one full nominal-frequency cycle.
    For 50 Hz and 6.4 kHz, one cycle is 128 samples.

    mode:
        real_imag      -> [Re(X_phasor), Im(X_phasor)]
        real_imag_mag  -> [Re, Im, abs]
    """
    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (N, T, C), got {X.shape}")

    N, T, C = X.shape

    if len(feature_names) != C:
        raise ValueError(
            f"feature_names length {len(feature_names)} does not match X channels {C}"
        )

    spc = int(np.rint(fs / f_nom))
    if spc <= 1:
        raise ValueError(f"Invalid samples per cycle: {spc}")

    if T < spc:
        raise ValueError(
            f"Window length T={T} is shorter than one cycle spc={spc}"
        )

    if step_samples <= 0:
        raise ValueError(f"step_samples must be positive, got {step_samples}")

    starts = np.arange(0, T - spc + 1, step_samples)

    if len(starts) == 0:
        raise ValueError("No phasor windows could be created.")

    n = np.arange(spc, dtype=np.float32)
    kernel = (2.0 / spc) * np.exp(-1j * 2.0 * np.pi * n / spc)

    phasor_steps = []

    for s in starts:
        segment = X[:, s : s + spc, :]  # (N, spc, C)
        ph = np.einsum("ntc,t->nc", segment, kernel)  # (N, C)
        phasor_steps.append(ph)

    phasors = np.stack(phasor_steps, axis=1)  # (N, S, C)

    if mode == "real_imag":
        X_out = np.concatenate(
            [phasors.real, phasors.imag],
            axis=-1,
        )
        out_names = (
            [f"phasor_real__{name}" for name in feature_names]
            + [f"phasor_imag__{name}" for name in feature_names]
        )

    elif mode == "real_imag_mag":
        X_out = np.concatenate(
            [phasors.real, phasors.imag, np.abs(phasors)],
            axis=-1,
        )
        out_names = (
            [f"phasor_real__{name}" for name in feature_names]
            + [f"phasor_imag__{name}" for name in feature_names]
            + [f"phasor_abs__{name}" for name in feature_names]
        )

    else:
        raise ValueError(
            f"Unknown phasor mode={mode!r}. Supported: real_imag, real_imag_mag"
        )

    return X_out.astype(np.float32), out_names


def apply_input_representation(
    *,
    X_used_filtered: np.ndarray,
    meta: dict[str, Any],
    feature_indices_for_ds: Any,
    config: MainConfig,
    logger,
) -> tuple[np.ndarray, dict[str, Any], Any]:
    """
    Applies the configured input representation.

    Supported:
        waveform   -> keep original waveform input
        phasor_seq -> convert waveform into sliding one-cycle phasor sequence

    Important:
        If feature_indices_for_ds is not None, this function first materializes
        those feature indices. After phasor conversion, feature_indices_for_ds
        is set to None because the feature axis has changed.
    """
    input_representation = str(
        getattr(config.training, "input_representation", "waveform")
    ).lower().strip()

    if input_representation == "waveform":
        return X_used_filtered, meta, feature_indices_for_ds

    logger.info("Input representation before transform: %s", X_used_filtered.shape)
    logger.info("Using input_representation=%s", input_representation)

    feature_names_full = list(meta["feature_names"])

    if feature_indices_for_ds is not None:
        feature_indices_list = list(feature_indices_for_ds)

        logger.info(
            "Materializing %d configured feature indices before representation transform.",
            len(feature_indices_list),
        )

        X_used_filtered = X_used_filtered[:, :, feature_indices_list]
        feature_names_for_x = [feature_names_full[i] for i in feature_indices_list]

        feature_indices_for_ds = None

    else:
        feature_names_for_x = feature_names_full

        if len(feature_names_for_x) != X_used_filtered.shape[-1]:
            logger.warning(
                "feature_names length (%d) does not match X feature dimension (%d). "
                "Using generic feature names for representation transform.",
                len(feature_names_for_x),
                X_used_filtered.shape[-1],
            )
            feature_names_for_x = [
                f"feature_{i}" for i in range(X_used_filtered.shape[-1])
            ]

    T_full = X_used_filtered.shape[1]
    window_s = float(config.window_extraction.window_length)
    fs = T_full / window_s

    logger.info(
        "Inferred fs=%.3f Hz from T=%d and window_s=%.6f",
        fs,
        T_full,
        window_s,
    )

    if input_representation == "phasor_seq":
        phasor_step_samples = int(
            getattr(config.training, "phasor_step_samples", 16)
        )
        phasor_mode = str(
            getattr(config.training, "phasor_mode", "real_imag")
        ).lower().strip()

        X_used_filtered, phasor_feature_names = build_sliding_phasor_sequence(
            X=X_used_filtered,
            feature_names=feature_names_for_x,
            fs=fs,
            f_nom=50.0,
            step_samples=phasor_step_samples,
            mode=phasor_mode,
        )

        meta["feature_names"] = phasor_feature_names

        logger.info(
            "Input representation after phasor transform: %s",
            X_used_filtered.shape,
        )
        logger.info("Number of phasor features: %d", len(phasor_feature_names))

        return X_used_filtered, meta, feature_indices_for_ds

    raise ValueError(
        f"Unknown training.input_representation={input_representation!r}. "
        f"Supported: waveform, phasor_seq"
    )
