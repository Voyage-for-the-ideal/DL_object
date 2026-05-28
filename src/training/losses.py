"""Loss, optimizer, and scheduler factories."""

from __future__ import annotations

from typing import Any


def get_torch_loss(name: str, **kwargs: Any) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for torch losses") from exc
    if name == "mse":
        return torch.nn.MSELoss()
    if name == "huber":
        delta = float(kwargs.get("huber_delta", 1.0))
        return torch.nn.HuberLoss(delta=delta)
    raise ValueError(f"Unsupported loss: {name}")


def get_torch_optimizer(name: str, parameters: Any, **kwargs: Any) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required") from exc
    lr = float(kwargs.get("learning_rate", 0.001))
    weight_decay = float(kwargs.get("weight_decay", 0.0))
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def get_torch_scheduler(name: str | None, optimizer: Any, **kwargs: Any) -> Any | None:
    if name is None or name == "none":
        return None
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required") from exc
    if name == "cosine_restart":
        t_0 = int(kwargs.get("scheduler_t0", 10))
        t_mult = int(kwargs.get("scheduler_t_mult", 1))
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=t_0, T_mult=t_mult
        )
    if name == "cosine":
        epochs = int(kwargs.get("epochs", 50))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "reduce_on_plateau":
        patience = int(kwargs.get("scheduler_patience", 5))
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=patience
        )
    raise ValueError(f"Unsupported scheduler: {name}")
