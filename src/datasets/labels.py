"""Forward return label construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schema import TRADE_DATE, TS_CODE, normalize_trade_date_column


def add_forward_return_labels(
    price: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 3, 5),
    drop_missing: bool = False,
) -> pd.DataFrame:
    """Add labels using close(T+h+1) / close(T+1) - 1 for each horizon h."""
    if price.empty:
        return pd.DataFrame(columns=[TRADE_DATE, TS_CODE, *[f"label_{h}d" for h in horizons]])
    frame = normalize_trade_date_column(price).copy()
    frame = frame.sort_values([TS_CODE, TRADE_DATE]).reset_index(drop=True)
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    grouped = frame.groupby(TS_CODE, group_keys=False)["close"]
    buy_close = grouped.shift(-1)
    for horizon in horizons:
        sell_close = grouped.shift(-(horizon + 1))
        frame[f"label_{horizon}d"] = sell_close / buy_close - 1
    labels = frame[[TRADE_DATE, TS_CODE, *[f"label_{h}d" for h in horizons]]].replace(
        [np.inf, -np.inf], np.nan
    )
    if drop_missing:
        labels = labels.dropna(subset=[f"label_{h}d" for h in horizons], how="all")
    return labels.reset_index(drop=True)


def merge_labels(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    return features.merge(labels, on=[TRADE_DATE, TS_CODE], how="inner")
