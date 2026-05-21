"""Model factory."""

from __future__ import annotations

from typing import Any

from src.models.base import BaseAlphaModel
from src.models.gbdt import GbdtRegressorModel
from src.models.linear import create_linear_model
from src.models.mlp import TorchMLPAlphaModel
from src.models.transformer import TorchTransformerAlphaModel


def create_model(name: str, input_dim: int, **kwargs: Any) -> BaseAlphaModel:
    if name in {"ridge", "elasticnet"}:
        return create_linear_model(name)
    if name in {"gbdt", "lightgbm"}:
        return GbdtRegressorModel()
    if name == "mlp":
        hidden_dim = int(kwargs.get("hidden_dim", 128))
        return TorchMLPAlphaModel(
            input_dim=input_dim,
            hidden_dims=(hidden_dim, max(hidden_dim // 2, 1)),
            dropout=float(kwargs.get("dropout", 0.1)),
            device=str(kwargs.get("device", "cpu")),
        )
    if name == "transformer_encoder":
        return TorchTransformerAlphaModel(
            input_dim=input_dim,
            hidden_dim=int(kwargs.get("hidden_dim", 128)),
            num_layers=int(kwargs.get("num_layers", 2)),
            num_heads=int(kwargs.get("num_heads", 4)),
            dropout=float(kwargs.get("dropout", 0.1)),
            device=str(kwargs.get("device", "cpu")),
        )
    raise ValueError(f"Unsupported model: {name}")
