from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from KOL.common.line_utils import get_line_params_for_row, get_default_side_tokens
from KOL.common.channel_mapping import extract_single_side_vi


class GroundK0Dataset(Dataset):
    def __init__(
        self,
        X_full: np.ndarray,
        labels_df: pd.DataFrame,
        y_all: np.ndarray,
        case_idx: np.ndarray,
        feature_names: list[str],
        topology: str,
    ):
        self.X_full = X_full
        self.labels_df = labels_df.reset_index(drop=True)
        self.y_all = y_all
        self.case_idx = case_idx
        self.feature_names = feature_names
        self.topology = str(topology)

    def __len__(self):
        return len(self.y_all)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        x_raw_full = np.asarray(self.X_full[idx], dtype=np.float32)

        bus_token, line_token = get_default_side_tokens(str(row["y_fault_line"]))
        x_vi = extract_single_side_vi(
            x_raw=x_raw_full,
            feature_names=self.feature_names,
            bus_token=bus_token,
            line_token=line_token,
        ).astype(np.float32)

        r1, x1, r0, x0, L_km = get_line_params_for_row(
            row=row,
            topology=self.topology,
        )

        return {
            "x_seq": torch.tensor(x_vi, dtype=torch.float32),
            "y": torch.tensor(float(self.y_all[idx]), dtype=torch.float32),
            "case_idx": torch.tensor(int(self.case_idx[idx]), dtype=torch.long),
            "r1": torch.tensor(r1, dtype=torch.float32),
            "x1": torch.tensor(x1, dtype=torch.float32),
            "r0": torch.tensor(r0, dtype=torch.float32),
            "x0": torch.tensor(x0, dtype=torch.float32),
            "line_len_km": torch.tensor(L_km, dtype=torch.float32),
            "dt_start": torch.tensor(float(row["dt_start"]), dtype=torch.float32),
            "sample_id": torch.tensor(int(row["sample_id"]), dtype=torch.long),
        }


def make_k0_loaders(
    X_used,
    labels_df_used,
    y_all,
    case_idx,
    idx_train,
    idx_val,
    idx_test,
    feature_names,
    topology: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
):
    ds_train = GroundK0Dataset(
        X_full=X_used[idx_train],
        labels_df=labels_df_used.iloc[idx_train].reset_index(drop=True),
        y_all=y_all[idx_train],
        case_idx=case_idx[idx_train],
        feature_names=feature_names,
        topology=topology,
    )
    ds_val = GroundK0Dataset(
        X_full=X_used[idx_val],
        labels_df=labels_df_used.iloc[idx_val].reset_index(drop=True),
        y_all=y_all[idx_val],
        case_idx=case_idx[idx_val],
        feature_names=feature_names,
        topology=topology,
    )
    ds_test = GroundK0Dataset(
        X_full=X_used[idx_test],
        labels_df=labels_df_used.iloc[idx_test].reset_index(drop=True),
        y_all=y_all[idx_test],
        case_idx=case_idx[idx_test],
        feature_names=feature_names,
        topology=topology,
    )

    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader
