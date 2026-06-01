"""Tree-model baseline with LightGBM fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.models.base import BaseAlphaModel


def _create_estimator(
    learning_rate: float = 0.05,
    n_estimators: int = 200,
    max_iter: int = 100,
    max_depth: int | None = None,
    random_state: int | None = 42,
) -> tuple[Any, str]:
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        kwargs = dict(
            max_iter=max_iter,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        if max_depth is not None:
            kwargs["max_depth"] = max_depth
        return (HistGradientBoostingRegressor(**kwargs), "hist_gradient_boosting")
    lgbm_kwargs: dict[str, Any] = dict(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    if max_depth is not None:
        lgbm_kwargs["max_depth"] = max_depth
    return (LGBMRegressor(**lgbm_kwargs), "lightgbm")


class GbdtRegressorModel(BaseAlphaModel):
    def __init__(
        self,
        estimator: Any | None = None,
        model_name: str | None = None,
        learning_rate: float = 0.05,
        n_estimators: int = 200,
        max_iter: int = 100,
        max_depth: int | None = None,
        random_state: int | None = 42,
    ) -> None:
        if estimator is None:
            estimator, fallback_name = _create_estimator(
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                max_iter=max_iter,
                max_depth=max_depth,
                random_state=random_state,
            )
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
