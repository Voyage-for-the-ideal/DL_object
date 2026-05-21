"""Historical window dataset for sequence models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.schema import TRADE_DATE, TS_CODE
from src.datasets.tabular_dataset import select_feature_columns


@dataclass
class WindowDataset:
    X: np.ndarray
    y: np.ndarray
    index: pd.DataFrame
    feature_columns: list[str]
    lookback: int
    label_column: str = "label_1d"

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        lookback: int,
        feature_columns: list[str] | None = None,
        label_column: str = "label_1d",
        drop_missing_history: bool = True,
    ) -> "WindowDataset":
        columns = feature_columns or select_feature_columns(frame, label_column)
        windows: list[np.ndarray] = []
        labels: list[float] = []
        index_rows: list[dict[str, str]] = []
        for ts_code, group in frame.sort_values([TS_CODE, TRADE_DATE]).groupby(TS_CODE):
            group = group.reset_index(drop=True)
            feature_values = group[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            label_values = pd.to_numeric(group[label_column], errors="coerce")
            for position in range(len(group)):
                start = position - lookback + 1
                if start < 0 and drop_missing_history:
                    continue
                if pd.isna(label_values.iloc[position]):
                    continue
                window = feature_values.iloc[max(0, start) : position + 1].to_numpy(
                    dtype=np.float32
                )
                if start < 0:
                    padding = np.zeros((-start, len(columns)), dtype=np.float32)
                    window = np.vstack([padding, window])
                windows.append(window)
                labels.append(float(label_values.iloc[position]))
                index_rows.append(
                    {TRADE_DATE: str(group.loc[position, TRADE_DATE]), TS_CODE: str(ts_code)}
                )
        X = (
            np.stack(windows).astype(np.float32)
            if windows
            else np.empty((0, lookback, len(columns)))
        )
        y = np.asarray(labels, dtype=np.float32)
        index = pd.DataFrame(index_rows, columns=[TRADE_DATE, TS_CODE])
        return cls(
            X=X,
            y=y,
            index=index,
            feature_columns=columns,
            lookback=lookback,
            label_column=label_column,
        )

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, item: int) -> tuple[np.ndarray, np.float32]:
        return self.X[item], self.y[item]
