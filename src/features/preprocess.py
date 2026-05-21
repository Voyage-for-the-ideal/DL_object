"""Train-only preprocessing with explicit fit/transform separation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.schema import TRADE_DATE


@dataclass
class TrainOnlyPreprocessor:
    winsorize: bool = True
    lower_quantile: float = 0.01
    upper_quantile: float = 0.99
    impute: bool = True
    scale: bool = True
    feature_columns: list[str] = field(default_factory=list)
    lower_bounds: pd.Series | None = None
    upper_bounds: pd.Series | None = None
    medians: pd.Series | None = None
    means: pd.Series | None = None
    stds: pd.Series | None = None
    fitted: bool = False

    def fit(
        self, frame: pd.DataFrame, feature_columns: list[str] | None = None
    ) -> "TrainOnlyPreprocessor":
        numeric = frame.select_dtypes(include=[np.number])
        self.feature_columns = feature_columns or list(numeric.columns)
        data = frame[self.feature_columns].apply(pd.to_numeric, errors="coerce")
        self.lower_bounds = data.quantile(self.lower_quantile)
        self.upper_bounds = data.quantile(self.upper_quantile)
        clipped = data.clip(lower=self.lower_bounds, upper=self.upper_bounds, axis=1)
        self.medians = clipped.median()
        filled = clipped.fillna(self.medians)
        self.means = filled.mean()
        self.stds = filled.std(ddof=0).replace(0, 1.0)
        self.fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fit before transform")
        result = frame.copy()
        data = result.reindex(columns=self.feature_columns).apply(pd.to_numeric, errors="coerce")
        if self.winsorize:
            data = data.clip(lower=self.lower_bounds, upper=self.upper_bounds, axis=1)
        if self.impute:
            data = data.fillna(self.medians)
        if self.scale:
            data = (data - self.means) / self.stds
        for column in self.feature_columns:
            result[column] = data[column]
        return result

    def fit_transform(
        self,
        frame: pd.DataFrame,
        feature_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        return self.fit(frame, feature_columns).transform(frame)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "TrainOnlyPreprocessor":
        return joblib.load(path)


def add_daily_rank_features(
    frame: pd.DataFrame,
    feature_columns: list[str],
    suffix: str = "_rank",
) -> pd.DataFrame:
    result = frame.copy()
    for column in feature_columns:
        if column in result.columns:
            result[f"{column}{suffix}"] = result.groupby(TRADE_DATE)[column].rank(pct=True)
    return result
