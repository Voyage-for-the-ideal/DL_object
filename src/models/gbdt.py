"""Tree-model baseline with LightGBM fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.models.base import BaseAlphaModel


def _create_estimator() -> tuple[Any, str]:
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        return (
            HistGradientBoostingRegressor(max_iter=100, learning_rate=0.05),
            "hist_gradient_boosting",
        )
    return LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42), "lightgbm"


class GbdtRegressorModel(BaseAlphaModel):
    def __init__(self, estimator: Any | None = None, model_name: str | None = None) -> None:
        if estimator is None:
            estimator, fallback_name = _create_estimator()
            model_name = model_name or fallback_name
        self.estimator = estimator
        self.model_name = model_name or "gbdt"

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "GbdtRegressorModel":
        self.estimator.fit(X, y)
        return self

    def predict_array(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.estimator.predict(X), dtype=float)

    def feature_importance(self, feature_columns: list[str]) -> pd.DataFrame:
        values = getattr(self.estimator, "feature_importances_", None)
        if values is None:
            return pd.DataFrame({"feature": feature_columns, "importance": np.nan})
        return pd.DataFrame({"feature": feature_columns, "importance": values})

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model_name": self.model_name, "estimator": self.estimator}, target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "GbdtRegressorModel":
        payload = joblib.load(path)
        return cls(payload["estimator"], payload["model_name"])
