"""Unified alpha model interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.schema import TRADE_DATE, TS_CODE


@dataclass
class ModelArtifact:
    model_name: str
    path: Path | None = None
    metrics: dict[str, float] | None = None
    metadata: dict[str, Any] | None = None


class BaseAlphaModel(ABC):
    model_name: str

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "BaseAlphaModel":
        raise NotImplementedError

    @abstractmethod
    def predict_array(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X: np.ndarray, index: pd.DataFrame) -> pd.DataFrame:
        return format_predictions(index, self.predict_array(X), self.model_name)

    @abstractmethod
    def save(self, path: str | Path) -> Path:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "BaseAlphaModel":
        raise NotImplementedError


def format_predictions(index: pd.DataFrame, scores: np.ndarray, model_name: str) -> pd.DataFrame:
    output = index[[TRADE_DATE, TS_CODE]].copy().reset_index(drop=True)
    output["score"] = np.asarray(scores).reshape(-1)
    output["model_name"] = model_name
    return output
