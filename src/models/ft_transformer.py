"""FT-Transformer: Feature Tokenizer + Transformer for tabular data.

Reference: Gorishniy et al. (2021) "Revisiting Deep Learning Models for Tabular Data"
https://arxiv.org/abs/2106.11959
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from src.models.base import BaseAlphaModel


class FeatureTokenizer(nn.Module):
    """Map each numeric feature to a d_token-dimensional embedding."""

    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Linear(1, d_token) for _ in range(n_features)
        ])
        self.n_features = n_features
        self.d_token = d_token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F]
        tokens = []
        for i, proj in enumerate(self.projections):
            tokens.append(proj(x[:, i:i+1]))  # [B, 1] -> [B, d_token]
        return torch.stack(tokens, dim=1)  # [B, F, d_token]


class DropPath(nn.Module):
    """Stochastic Depth (DropPath)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x / keep_prob * random_tensor


class FTTransformerBlock(nn.Module):
    """Pre-LN Transformer block with DropPath."""

    def __init__(self, d_token: int, n_heads: int, ffn_ratio: int,
                 dropout: float, drop_path: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(
            d_token, n_heads, dropout=dropout, batch_first=True
        )
        self.drop_path1 = DropPath(drop_path)
        self.ln2 = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_token * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token * ffn_ratio, d_token),
            nn.Dropout(dropout),
        )
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN + MHA + DropPath + Residual
        x = x + self.drop_path1(self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0])
        # Pre-LN + FFN + DropPath + Residual
        x = x + self.drop_path2(self.ffn(self.ln2(x)))
        return x


class FTTransformer(nn.Module):
    """FT-Transformer for tabular regression.

    Args:
        n_features: Number of input numeric features.
        d_token: Token embedding dimension.
        n_blocks: Number of Transformer blocks.
        n_heads: Number of attention heads.
        ffn_ratio: FFN hidden dim = d_token * ffn_ratio.
        dropout: Attention and FFN dropout rate.
        drop_path: Maximum DropPath rate (linearly increased across blocks).
    """

    def __init__(
        self,
        n_features: int,
        d_token: int = 128,
        n_blocks: int = 4,
        n_heads: int = 8,
        ffn_ratio: int = 2,
        dropout: float = 0.2,
        drop_path: float = 0.1,
    ):
        super().__init__()
        if d_token % n_heads != 0:
            raise ValueError(f"d_token ({d_token}) must be divisible by n_heads ({n_heads})")

        self.tokenizer = FeatureTokenizer(n_features, d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        # Optional learnable feature positional encoding
        self.feat_pe = nn.Parameter(torch.zeros(1, n_features + 1, d_token))

        # Build Transformer blocks with linearly increasing DropPath
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            dp = (i / max(n_blocks - 1, 1)) * drop_path
            self.blocks.append(
                FTTransformerBlock(
                    d_token=d_token,
                    n_heads=n_heads,
                    ffn_ratio=ffn_ratio,
                    dropout=dropout,
                    drop_path=dp,
                )
            )

        self.head_ln = nn.LayerNorm(d_token)
        self.head = nn.Sequential(
            nn.Linear(d_token, d_token // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_token // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.feat_pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F]
        tokens = self.tokenizer(x)  # [B, F, d_token]
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # [B, 1, d_token]
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # [B, F+1, d_token]
        tokens = tokens + self.feat_pe

        for block in self.blocks:
            tokens = block(tokens)

        cls_out = tokens[:, 0, :]  # [B, d_token]
        cls_out = self.head_ln(cls_out)
        return self.head(cls_out).squeeze(-1)  # [B]


class FTTransformerAlphaModel(BaseAlphaModel):
    """Wraps FTTransformer following the BaseAlphaModel interface."""

    def __init__(
        self,
        n_features: int,
        d_token: int = 128,
        n_blocks: int = 4,
        n_heads: int = 8,
        ffn_ratio: int = 2,
        dropout: float = 0.2,
        drop_path: float = 0.1,
        device: str = "cpu",
    ):
        self.model = FTTransformer(
            n_features=n_features,
            d_token=d_token,
            n_blocks=n_blocks,
            n_heads=n_heads,
            ffn_ratio=ffn_ratio,
            dropout=dropout,
            drop_path=drop_path,
        )
        self.device = device
        self.model_name = "ft_transformer"
        self.model.to(device)
        self._model_kwargs = {
            "n_features": n_features, "d_token": d_token,
            "n_blocks": n_blocks, "n_heads": n_heads, "ffn_ratio": ffn_ratio,
            "dropout": dropout, "drop_path": drop_path,
        }

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        raise NotImplementedError("Use train_tabular_model() instead")

    def predict_array(self, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        self.model.eval()
        scores: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                batch = torch.as_tensor(X[start : start + batch_size], dtype=torch.float32, device=self.device)
                scores.append(self.model(batch).detach().cpu().numpy())
        return np.concatenate(scores)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state_dict": self.model.state_dict(), "kwargs": self._model_kwargs},
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "FTTransformerAlphaModel":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        kwargs = checkpoint["kwargs"]
        instance = cls(**kwargs)
        instance.model.load_state_dict(checkpoint["model_state_dict"])
        instance.model.to(instance.device)
        return instance
