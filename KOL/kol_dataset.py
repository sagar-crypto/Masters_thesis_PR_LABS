from __future__ import annotations
import numpy as np
import torch
from typing import Any, cast
from collections.abc import Sized
from torch.utils.data import Dataset
from .operator_bank import KnownOperatorBank

class KOLAugmentedDataset(Dataset):
    """
    Wraps an existing dataset that returns (x, y) with x: (T,F).
    Produces x_aug: (T, F+1) where last channel is constant = d_phys.
    """

    def __init__(
        self,
        base_ds: Any,
        op_bank: KnownOperatorBank,
        return_ops: bool = False,
    ):
        self.base_ds = base_ds
        self.op_bank = op_bank
        self.return_ops = bool(return_ops)

    def __len__(self) -> int:
        return len(cast(Sized, self.base_ds))

    def __getitem__(self, idx: int):
        x, y = self.base_ds[idx]  # x: torch (T,F)
        x_np = x.detach().cpu().numpy()
        ops, d_phys = self.op_bank.compute(x_np)

        T = x.shape[0]
        phys_chan = torch.full((T, 1), float(d_phys), dtype=x.dtype)
        x_aug = torch.cat([x, phys_chan], dim=1)

        if self.return_ops:
            return x_aug, y, torch.from_numpy(ops), torch.tensor(d_phys, dtype=torch.float32)
        return x_aug, y
