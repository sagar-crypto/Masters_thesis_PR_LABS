from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SequenceWithPriorCaseDataset(Dataset):
    def __init__(self, X, y, d_phys_prior, case_idx, op_features=None, feature_indices=None):
        self.X = X
        self.y = y
        self.d_phys_prior = d_phys_prior
        self.case_idx = case_idx
        self.op_features = op_features
        self.feature_indices = (
            np.asarray(feature_indices, dtype=np.int64)
            if feature_indices is not None
            else None
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_np = self.X[idx]
        if self.feature_indices is not None:
            x_np = x_np[:, self.feature_indices]

        x = torch.tensor(np.asarray(x_np), dtype=torch.float32)
        d_prior = torch.tensor(self.d_phys_prior[idx], dtype=torch.float32)
        c_idx = torch.tensor(self.case_idx[idx], dtype=torch.long)

        if self.op_features is None:
            op_feat = torch.zeros(0, dtype=torch.float32)
        else:
            op_feat = torch.tensor(self.op_features[idx], dtype=torch.float32)

        y_val = self.y[idx]
        if np.issubdtype(np.asarray(self.y).dtype, np.integer):
            y = torch.tensor(y_val, dtype=torch.long)
        else:
            y = torch.tensor(y_val, dtype=torch.float32)

        return x, d_prior, c_idx, op_feat, y


def make_kol_loaders(
    X_used,
    y_all,
    d_phys_prior,
    case_idx,
    op_features,
    idx_train,
    idx_val,
    idx_test,
    feature_indices_for_ds,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
):
    ds_train = SequenceWithPriorCaseDataset(
        X_used[idx_train], y_all[idx_train], d_phys_prior[idx_train], case_idx[idx_train],
        op_features=None if op_features is None else op_features[idx_train],
        feature_indices=feature_indices_for_ds,
    )
    ds_val = SequenceWithPriorCaseDataset(
        X_used[idx_val], y_all[idx_val], d_phys_prior[idx_val], case_idx[idx_val],
        op_features=None if op_features is None else op_features[idx_val],
        feature_indices=feature_indices_for_ds,
    )
    ds_test = SequenceWithPriorCaseDataset(
        X_used[idx_test], y_all[idx_test], d_phys_prior[idx_test], case_idx[idx_test],
        op_features=None if op_features is None else op_features[idx_test],
        feature_indices=feature_indices_for_ds,
    )

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader

class KOLDualInputDataset(Dataset):
    """
    Dataset for waveform + phasor dual-input KOL residual model.

    Each sample returns:
        x_waveform
        x_phasor
        y
        d_prior
        case_idx
        op_features
    """

    def __init__(
        self,
        *,
        X_waveform: np.ndarray,
        X_phasor: np.ndarray,
        y_all: np.ndarray,
        d_phys_prior: np.ndarray,
        case_idx: np.ndarray,
        op_features: np.ndarray | None,
        indices: np.ndarray,
    ) -> None:
        self.X_waveform = X_waveform
        self.X_phasor = X_phasor
        self.y_all = y_all
        self.d_phys_prior = d_phys_prior
        self.case_idx = case_idx
        self.op_features = op_features
        self.indices = np.asarray(indices, dtype=int)

        if len(self.X_waveform) != len(self.X_phasor):
            raise ValueError(
                f"X_waveform and X_phasor must have same N, got "
                f"{len(self.X_waveform)} and {len(self.X_phasor)}"
            )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        idx = int(self.indices[item])

        x_waveform = torch.as_tensor(self.X_waveform[idx], dtype=torch.float32)
        x_phasor = torch.as_tensor(self.X_phasor[idx], dtype=torch.float32)
        y = torch.as_tensor(self.y_all[idx], dtype=torch.float32)
        d_prior = torch.as_tensor(self.d_phys_prior[idx], dtype=torch.float32)
        case = torch.as_tensor(self.case_idx[idx], dtype=torch.long)

        if self.op_features is None:
            op = torch.empty(0, dtype=torch.float32)
        else:
            op = torch.as_tensor(self.op_features[idx], dtype=torch.float32)

        return x_waveform, x_phasor, y, d_prior, case, op


def make_kol_dual_loaders(
    *,
    X_waveform: np.ndarray,
    X_phasor: np.ndarray,
    y_all: np.ndarray,
    d_phys_prior: np.ndarray,
    case_idx: np.ndarray,
    op_features: np.ndarray | None,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
):
    train_ds = KOLDualInputDataset(
        X_waveform=X_waveform,
        X_phasor=X_phasor,
        y_all=y_all,
        d_phys_prior=d_phys_prior,
        case_idx=case_idx,
        op_features=op_features,
        indices=idx_train,
    )

    val_ds = KOLDualInputDataset(
        X_waveform=X_waveform,
        X_phasor=X_phasor,
        y_all=y_all,
        d_phys_prior=d_phys_prior,
        case_idx=case_idx,
        op_features=op_features,
        indices=idx_val,
    )

    test_ds = KOLDualInputDataset(
        X_waveform=X_waveform,
        X_phasor=X_phasor,
        y_all=y_all,
        d_phys_prior=d_phys_prior,
        case_idx=case_idx,
        op_features=op_features,
        indices=idx_test,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
