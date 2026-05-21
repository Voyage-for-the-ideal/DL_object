"""Transformer encoder regression model."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from src.models.base import BaseAlphaModel

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None


if torch is not None:

    class PositionalEncoding(nn.Module):
        def __init__(self, hidden_dim: int, max_len: int = 512) -> None:
            super().__init__()
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))
            pe = torch.zeros(max_len, hidden_dim)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.pe[:, : x.size(1)]

    class TransformerRegressor(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 128,
            num_layers: int = 2,
            num_heads: int = 4,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.projection = nn.Linear(input_dim, hidden_dim)
            self.position = PositionalEncoding(hidden_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            hidden = self.position(self.projection(x))
            encoded = self.encoder(hidden)
            return self.head(encoded[:, -1, :]).squeeze(-1)

else:

    class TransformerRegressor:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch is required for TransformerRegressor")


class TorchTransformerAlphaModel(BaseAlphaModel):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        device: str = "cpu",
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch is required for TorchTransformerAlphaModel")
        self.model_name = "transformer_encoder"
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.device = torch.device(device)
        self.model = TransformerRegressor(input_dim, hidden_dim, num_layers, num_heads, dropout).to(
            self.device
        )

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "TorchTransformerAlphaModel":
        epochs = int(kwargs.get("epochs", 10))
        batch_size = int(kwargs.get("batch_size", 256))
        learning_rate = float(kwargs.get("learning_rate", 0.001))
        dataset = torch.utils.data.TensorDataset(
            torch.as_tensor(X, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.float32),
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()
        self.model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()
        return self

    def predict_array(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            return self.model(tensor).detach().cpu().numpy()

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_name": self.model_name,
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
                "state_dict": self.model.state_dict(),
            },
            target,
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "TorchTransformerAlphaModel":
        if torch is None:
            raise ImportError("PyTorch is required for TorchTransformerAlphaModel")
        payload = torch.load(path, map_location="cpu")
        model = cls(
            input_dim=int(payload["input_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            num_layers=int(payload["num_layers"]),
            num_heads=int(payload["num_heads"]),
            dropout=float(payload["dropout"]),
            device="cpu",
        )
        model.model.load_state_dict(payload["state_dict"])
        model.model.eval()
        return model
