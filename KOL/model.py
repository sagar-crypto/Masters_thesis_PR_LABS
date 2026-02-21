import torch
import torch.nn as nn
from typing import Tuple, Union

class WaveDeltaCNN(nn.Module):
    """
    Predicts delta in normalized space (0–1) to correct the classical estimate:
        y_hat_n = cb_n + delta_n
    """
    def __init__(self, n_channels: int, context_dim: int, dropout: float = 0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # -> (B,128,1)
        )

        self.head = nn.Sequential(
            nn.Linear(128 + context_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x_win: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x_win: (B,T,C) -> Conv1d expects (B,C,T)
        x = x_win.transpose(1, 2)              # (B,C,T)
        z = self.cnn(x).squeeze(-1)            # (B,128)
        h = torch.cat([z, context], dim=1)     # (B,128+context_dim)
        delta = self.head(h).squeeze(-1)       # (B,)
        return delta
    

class MLPBlock(nn.Module):
    def __init__(self, d: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, d * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d * mlp_ratio, d),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        return x + self.ff(self.ln(x))

class WaveDeltaPatchMLP(nn.Module):
    """
    Pure-MLP time-series model:
    - patchify over time
    - per-patch embedding (Linear)
    - deep residual MLP blocks (LayerNorm + GELU)
    - output delta (optionally gate later)

    Input: x_win (B,T,C), context (B,Dctx)
    """
    def __init__(
        self,
        n_channels: int,
        context_dim: int,
        seq_len: int,
        patch_len: int = 8,
        d_model: int = 256,
        depth: int = 96,          # try 48 first; 96 if stable
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        use_gate: bool = False,
    ):
        super().__init__()
        assert seq_len % patch_len == 0, f"seq_len={seq_len} must be divisible by patch_len={patch_len}"
        self.patch_len = patch_len
        self.n_patches = seq_len // patch_len
        self.use_gate = use_gate

        patch_dim = n_channels * patch_len  # flatten C*patch_len
        self.patch_embed = nn.Sequential(
            nn.Linear(patch_dim, d_model),
            nn.GELU(),
        )

        # context -> FiLM conditioning (simple and effective)
        self.ctx_to_scale = nn.Linear(context_dim, d_model)
        self.ctx_to_shift = nn.Linear(context_dim, d_model)

        self.blocks = nn.Sequential(*[
            MLPBlock(d_model, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        self.ln_out = nn.LayerNorm(d_model)

        out_dim = 2 if use_gate else 1
        self.head = nn.Linear(d_model, out_dim)

        # gate bias init (prefer cb initially) if gate is enabled
        if use_gate:
            with torch.no_grad():
                self.head.bias.data[1] = 3.0  # sigmoid(3) ~ 0.95

    def forward(self, x_win: torch.Tensor, context: torch.Tensor):
        # x_win: (B,T,C)
        B, T, C = x_win.shape

        # patchify: (B, n_patches, patch_len, C) -> (B, n_patches, patch_len*C)
        x = x_win.view(B, self.n_patches, self.patch_len, C).reshape(B, self.n_patches, self.patch_len * C)

        x = self.patch_embed(x)  # (B, N, D)

        # FiLM: condition each patch embedding on context
        scale = self.ctx_to_scale(context).unsqueeze(1)  # (B,1,D)
        shift = self.ctx_to_shift(context).unsqueeze(1)  # (B,1,D)
        x = x * (1.0 + torch.tanh(scale)) + shift

        x = self.blocks(x)               # (B,N,D)
        x = self.ln_out(x).mean(dim=1)   # global average over patches -> (B,D)

        out = self.head(x)               # (B,1) or (B,2)

        if self.use_gate:
            delta = out[:, 0]
            gate_logit = out[:, 1]
            delta = 0.5 * torch.tanh(delta)  # constrain delta
            return delta, gate_logit

        delta = out[:, 0]
        delta = 0.5 * torch.tanh(delta)
        return delta
