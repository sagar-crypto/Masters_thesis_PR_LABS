from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .phasor_utils import phasors_for_channels

@dataclass(frozen=True)
class OperatorBankConfig:
    fs: float                 # sampling frequency (Hz)
    f0: float = 50.0          # fundamental (Hz) - set 50 in DE, 60 in US
    # indices into feature dimension F for V/I channels
    v_idx: tuple[int, ...] = (0, 1, 2)   # e.g., Va,Vb,Vc
    i_idx: tuple[int, ...] = (3, 4, 5)   # e.g., Ia,Ib,Ic

    # line impedance magnitude (Ohm) for full line length, OR per-km with length_km
    z_line_ohm_per_km: float | None = None
    length_km: float | None = None
    z_line_total_ohm: float | None = None

    eps: float = 1e-9


class KnownOperatorBank:
    """
    Minimal KOL operator bank:
      waveform window -> phasors -> operators -> physics distance estimate
    """

    def __init__(self, cfg: OperatorBankConfig):
        self.cfg = cfg

        if cfg.z_line_total_ohm is None:
            if cfg.z_line_ohm_per_km is None or cfg.length_km is None:
                # still usable: we will output a *proxy* distance in arbitrary units
                self._z_line_mag = None
            else:
                self._z_line_mag = abs(cfg.z_line_ohm_per_km * cfg.length_km)
        else:
            self._z_line_mag = abs(cfg.z_line_total_ohm)

    def compute(self, X_win: np.ndarray) -> tuple[np.ndarray, float]:
        """
        X_win: (T,F)
        Returns:
          ops: (D_ops,) numeric operator features
          d_phys: physics distance estimate in [0,1] if line impedance known,
                  else a proxy (positive scalar)
        """
        cfg = self.cfg
        V = phasors_for_channels(X_win, fs=cfg.fs, f0=cfg.f0, ch_idx=list(cfg.v_idx))  # (nV,)
        I = phasors_for_channels(X_win, fs=cfg.fs, f0=cfg.f0, ch_idx=list(cfg.i_idx))  # (nI,)

        # simple aggregates (robust for first prototype)
        Vmag = np.abs(V)
        Imag = np.abs(I)

        # apparent impedance per phase (avoid divide-by-zero)
        Z = V / (I + cfg.eps)
        Zmag = np.abs(Z)

        # operator vector: [Vmag stats, Imag stats, Zmag stats, angle diffs]
        ops = np.array(
            [
                Vmag.mean(), Vmag.std(),
                Imag.mean(), Imag.std(),
                Zmag.mean(), Zmag.std(),
                np.angle(V).mean(), np.angle(I).mean(),
                (np.angle(V) - np.angle(I)).mean(),
            ],
            dtype=np.float32,
        )

        # physics distance: normalize by known line impedance magnitude if available
        z_app = float(np.median(Zmag))  # robust single scalar
        if self._z_line_mag is None or self._z_line_mag < cfg.eps:
            d_phys = max(cfg.eps, z_app)  # proxy (not normalized)
        else:
            d_phys = float(np.clip(z_app / self._z_line_mag, 0.0, 1.0))

        return ops, d_phys
