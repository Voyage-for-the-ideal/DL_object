"""Date-based train/validation splitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.data.schema import TRADE_DATE, normalize_trade_date


@dataclass(frozen=True)
class TimeSplit:
    train_start: str
    train_end: str
    valid_start: str
    valid_end: str

    def normalized(self) -> "TimeSplit":
        return TimeSplit(
            normalize_trade_date(self.train_start),
            normalize_trade_date(self.train_end),
            normalize_trade_date(self.valid_start),
            normalize_trade_date(self.valid_end),
        )


def split_by_date(frame: pd.DataFrame, split: TimeSplit) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = split.normalized()
    data = frame.copy()
    data[TRADE_DATE] = data[TRADE_DATE].map(normalize_trade_date)
    train = data[(data[TRADE_DATE] >= spec.train_start) & (data[TRADE_DATE] <= spec.train_end)]
    valid = data[(data[TRADE_DATE] >= spec.valid_start) & (data[TRADE_DATE] <= spec.valid_end)]
    return train.reset_index(drop=True), valid.reset_index(drop=True)


def split_from_config(
    frame: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data: dict[str, Any] = config["data"] if isinstance(config.get("data"), dict) else {}
    split = TimeSplit(
        train_start=str(data.get("start_date")),
        train_end=str(data.get("train_end_date")),
        valid_start=str(data.get("valid_start_date")),
        valid_end=str(data.get("valid_end_date")),
    )
    return split_by_date(frame, split)
