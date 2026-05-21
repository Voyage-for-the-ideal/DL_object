"""Loss factory."""

from __future__ import annotations

from typing import Any


def get_torch_loss(name: str) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for torch losses") from exc
    if name == "mse":
        return torch.nn.MSELoss()
    if name == "huber":
        return torch.nn.HuberLoss()
    raise ValueError(f"Unsupported loss: {name}")
