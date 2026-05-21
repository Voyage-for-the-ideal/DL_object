"""Tabular datasets for linear, tree, and MLP models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.data.schema import TRADE_DATE, TS_CODE

EXCLUDED_PREFIXES = ("label_", "future_", "target_")
EXCLUDED_COLUMNS = {
    TRADE_DATE,
    TS_CODE,
    "signal_date",
    "next_trade_date",
    "label_end_date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
}


def select_feature_columns(frame: pd.DataFrame, label_column: str = "label_1d") -> list[str]:
    columns: list[str] = []
    for column in frame.columns:
        if column == label_column or column in EXCLUDED_COLUMNS:
            continue
        if any(column.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(column)
    return columns


@dataclass
class TabularDataset:
    X: np.ndarray
    y: np.ndarray
    index: pd.DataFrame
    feature_columns: list[str]
    label_column: str = "label_1d"

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        feature_columns: list[str] | None = None,
        label_column: str = "label_1d",
        dropna_label: bool = True,
    ) -> "TabularDataset":
        data = frame.copy()
        if dropna_label:
            data = data.dropna(subset=[label_column])
        columns = feature_columns or select_feature_columns(data, label_column)
        X = (
            data[columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
        y = (
            pd.to_numeric(data[label_column], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
        index = data[[TRADE_DATE, TS_CODE]].reset_index(drop=True)
        return cls(X=X, y=y, index=index, feature_columns=columns, label_column=label_column)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, item: int) -> tuple[np.ndarray, np.float32]:
        return self.X[item], self.y[item]

    def to_sklearn(self) -> tuple[np.ndarray, np.ndarray]:
        return self.X, self.y


def make_torch_dataloader(
    dataset: TabularDataset,
    batch_size: int,
    shuffle: bool = False,
) -> Any:
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError("PyTorch is required for torch DataLoader construction") from exc
    tensor_dataset = TensorDataset(torch.from_numpy(dataset.X), torch.from_numpy(dataset.y))
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)
