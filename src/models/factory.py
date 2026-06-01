"""Model factory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.models.base import BaseAlphaModel
from src.models.ft_transformer import FTTransformerAlphaModel
from src.models.gbdt import GbdtRegressorModel
from src.models.linear import SklearnRegressorModel, create_linear_model
from src.models.mlp import TorchMLPAlphaModel
from src.models.transformer import TorchTransformerAlphaModel


def load_model_artifact(path: str | Path, model_name: str) -> BaseAlphaModel:
    if model_name in {"ridge", "elasticnet"}:
        return SklearnRegressorModel.load(path)
    if model_name in {"gbdt", "lightgbm", "hist_gradient_boosting"}:
        return GbdtRegressorModel.load(path)
    if model_name == "mlp":
        return TorchMLPAlphaModel.load(path)
    if model_name == "transformer_encoder":
        return TorchTransformerAlphaModel.load(path)
    if model_name == "ft_transformer":
        return FTTransformerAlphaModel.load(path)
    raise ValueError(f"Unsupported model artifact type: {model_name}")


def create_model(name: str, input_dim: int, **kwargs: Any) -> BaseAlphaModel:
    if name in {"ridge", "elasticnet"}:
        return create_linear_model(name, **kwargs)
    if name in {"gbdt", "lightgbm"}:
        return GbdtRegressorModel(**kwargs)
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
            lookback=int(kwargs.get("lookback", 20)),
            device=str(kwargs.get("device", "cpu")),
        )
    if name == "ft_transformer":
        # Merge nested ft_transformer sub-config so that the YAML
        #   model.ft_transformer.{d_token,n_blocks,...}  takes effect.
        ftt_kwargs = dict(kwargs)
        if "ft_transformer" in ftt_kwargs:
            ftt_kwargs.update(ftt_kwargs.pop("ft_transformer"))
        return FTTransformerAlphaModel(
            n_features=input_dim,
            d_token=int(ftt_kwargs.get("d_token", 128)),
            n_blocks=int(ftt_kwargs.get("n_blocks", 4)),
            n_heads=int(ftt_kwargs.get("n_heads", 8)),
            ffn_ratio=int(ftt_kwargs.get("ffn_ratio", 2)),
            dropout=float(ftt_kwargs.get("dropout", 0.2)),
            drop_path=float(ftt_kwargs.get("drop_path", 0.1)),
            device=str(ftt_kwargs.get("device", "cpu")),
        )
    raise ValueError(f"Unsupported model: {name}")
