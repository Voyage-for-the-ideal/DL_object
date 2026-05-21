"""Fundamental and valuation feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schema import METRIC_COLUMNS, TRADE_DATE, TS_CODE, normalize_trade_date_column


def build_metric_features(metric: pd.DataFrame) -> pd.DataFrame:
    if metric.empty:
        return pd.DataFrame(columns=[TRADE_DATE, TS_CODE])
    frame = normalize_trade_date_column(metric).copy()
    result = frame[[TRADE_DATE, TS_CODE]].copy()
    for column in METRIC_COLUMNS:
        if column in frame.columns:
            result[f"metric_{column}"] = pd.to_numeric(frame[column], errors="coerce")
    for column in ("total_mv", "circ_mv"):
        metric_column = f"metric_{column}"
        if metric_column in result.columns:
            value = pd.to_numeric(result[metric_column], errors="coerce")
            result[f"{metric_column}_log"] = np.log1p(value.clip(lower=0))
    return result.replace([np.inf, -np.inf], np.nan)
