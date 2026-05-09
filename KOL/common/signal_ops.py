from __future__ import annotations

import numpy as np
import torch


def dft_phasor_1cycle(x: np.ndarray) -> complex:
    n = len(x)
    if n <= 0:
        return 0j
    k = np.arange(n, dtype=np.float64)
    w = np.exp(-1j * 2.0 * np.pi * k / n)
    return (2.0 / n) * np.sum(x.astype(np.float64) * w)


def symm_pos_seq(a: complex, b: complex, c: complex) -> complex:
    alpha = np.exp(1j * 2.0 * np.pi / 3.0)
    return (a + alpha * b + (alpha ** 2) * c) / 3.0


def symm_zero_seq(a: complex, b: complex, c: complex) -> complex:
    return (a + b + c) / 3.0


def symm_neg_seq(a: complex, b: complex, c: complex) -> complex:
    alpha = np.exp(1j * 2.0 * np.pi / 3.0)
    return (a + (alpha ** 2) * b + alpha * c) / 3.0


def dft_phasor_1cycle_torch(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]
    if n <= 0:
        return torch.zeros(x.shape[:-1], dtype=torch.complex64, device=x.device)

    k = torch.arange(n, dtype=torch.float32, device=x.device)
    w = torch.exp(-1j * 2.0 * torch.pi * k / n).to(torch.complex64)

    x_c = x.to(torch.complex64)
    return (2.0 / n) * torch.sum(x_c * w, dim=-1)


def classical_alpha_torch(device: torch.device) -> torch.Tensor:
    return torch.tensor(
        complex(-0.5, np.sqrt(3.0) / 2.0),
        dtype=torch.complex64,
        device=device,
    )
