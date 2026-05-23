"""Linear sklearn baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import ElasticNet, Ridge

from src.models.base import BaseAlphaModel


class SklearnRegressorModel(BaseAlphaModel):
    def __init__(self, estimator: Any, model_name: str) -> None:
        self.estimator = estimator
        self.model_name = model_name

    @classmethod
    def ridge(cls, alpha: float = 1.0) -> "SklearnRegressorModel":
        return cls(Ridge(alpha=alpha), "ridge")

    @classmethod
    def elasticnet(
        cls,
        alpha: float = 0.00001,
        l1_ratio: float = 0.5,
        max_iter: int = 10000,
    ) -> "SklearnRegressorModel":
        return cls(
            ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter),
            "elasticnet",
        )

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "SklearnRegressorModel":
        self.estimator.fit(X, y)
        return self

    def predict_array(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.estimator.predict(X), dtype=float)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model_name": self.model_name, "estimator": self.estimator}, target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "SklearnRegressorModel":
        payload = joblib.load(path)
        return cls(payload["estimator"], payload["model_name"])


def create_linear_model(name: str, **params: Any) -> SklearnRegressorModel:
    if name == "ridge":
        return SklearnRegressorModel.ridge(**params)
    if name == "elasticnet":
        return SklearnRegressorModel.elasticnet(**params)
    raise ValueError(f"Unsupported linear model: {name}")
