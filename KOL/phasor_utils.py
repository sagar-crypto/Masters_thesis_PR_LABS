from __future__ import annotations
import numpy as np

def phasor_dft_at_f0(x: np.ndarray, fs: float, f0: float) -> complex:
    """
    Compute single-frequency DFT coefficient at f0 for a real signal x[t].
    Returns complex phasor proportional to amplitude and phase.

    x: shape (T,)
    """
    x = np.asarray(x, dtype=np.float64)
    T = x.shape[0]
    n = np.arange(T, dtype=np.float64)
    w = np.exp(-1j * 2.0 * np.pi * f0 * n / fs)
    # scale: 2/T makes it closer to amplitude for a pure sinusoid
    return (2.0 / max(1, T)) * np.sum(x * w)


def phasors_for_channels(X_win: np.ndarray, fs: float, f0: float, ch_idx: list[int]) -> np.ndarray:
    """
    X_win: (T, F)
    returns complex phasors: (len(ch_idx),)
    """
    ph = []
    for j in ch_idx:
        ph.append(phasor_dft_at_f0(X_win[:, j], fs=fs, f0=f0))
    return np.asarray(ph, dtype=np.complex128)
