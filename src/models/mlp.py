"""PyTorch MLP model."""

from __future__ import annotations

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

    class MLPRegressor(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_dims: tuple[int, ...] = (128, 64),
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            last_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.extend(
                    [
                        nn.Linear(last_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    ]
                )
                last_dim = hidden_dim
            layers.append(nn.Linear(last_dim, 1))
            self.network = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x).squeeze(-1)

else:

    class MLPRegressor:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch is required for MLPRegressor")


class TorchMLPAlphaModel(BaseAlphaModel):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.1,
        device: str = "cpu",
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch is required for TorchMLPAlphaModel")
        self.model_name = "mlp"
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.device = torch.device(device)
        self.model = MLPRegressor(input_dim, hidden_dims, dropout).to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "TorchMLPAlphaModel":
        epochs = int(kwargs.get("epochs", 10))
        batch_size = int(kwargs.get("batch_size", 1024))
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
                "hidden_dims": self.hidden_dims,
                "dropout": self.dropout,
                "state_dict": self.model.state_dict(),
            },
            target,
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "TorchMLPAlphaModel":
        if torch is None:
            raise ImportError("PyTorch is required for TorchMLPAlphaModel")
        payload = torch.load(path, map_location="cpu")
        model = cls(
            input_dim=int(payload["input_dim"]),
            hidden_dims=tuple(payload["hidden_dims"]),
            dropout=float(payload["dropout"]),
            device="cpu",
        )
        model.model.load_state_dict(payload["state_dict"])
        model.model.eval()
        return model
