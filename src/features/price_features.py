"""Leakage-safe price and volume feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schema import TRADE_DATE, TS_CODE, normalize_trade_date_column


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return (numerator / denominator).replace([np.inf, -np.inf], np.nan)


def build_price_features(
    daily: pd.DataFrame,
    windows: tuple[int, ...] = (5, 10, 20),
) -> pd.DataFrame:
    """Build features whose rolling windows end at the signal date ``T``."""
    if daily.empty:
        return pd.DataFrame(columns=[TRADE_DATE, TS_CODE])
    frame = normalize_trade_date_column(daily).copy()
    frame = frame.sort_values([TS_CODE, TRADE_DATE]).reset_index(drop=True)
    numeric_columns = ["open", "high", "low", "close", "pre_close", "vol", "amount", "vwap"]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    features = frame[[TRADE_DATE, TS_CODE]].copy()
    features["ret_1d"] = _safe_divide(frame["close"], frame["pre_close"]) - 1
    features["oc_ret"] = _safe_divide(frame["close"], frame["open"]) - 1
    features["hl_range"] = _safe_divide(frame["high"], frame["low"]) - 1
    features["vwap_deviation"] = _safe_divide(frame["close"], frame["vwap"]) - 1
    features["volume_chg"] = frame.groupby(TS_CODE)["vol"].pct_change()
    features["amount_chg"] = frame.groupby(TS_CODE)["amount"].pct_change()

    grouped_close = frame.groupby(TS_CODE, group_keys=False)["close"]
    grouped_ret = features.groupby(frame[TS_CODE], group_keys=False)["ret_1d"]
    grouped_vol = frame.groupby(TS_CODE, group_keys=False)["vol"]
    grouped_amount = frame.groupby(TS_CODE, group_keys=False)["amount"]

    for window in windows:
        features[f"ret_mean_{window}"] = (
            grouped_ret.rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        features[f"ret_vol_{window}"] = (
            grouped_ret.rolling(window, min_periods=2).std().reset_index(level=0, drop=True)
        )
        shifted = grouped_close.shift(window)
        features[f"momentum_{window}"] = _safe_divide(frame["close"], shifted) - 1
        features[f"ma_close_{window}"] = (
            grouped_close.rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        features[f"vol_mean_{window}"] = (
            grouped_vol.rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        features[f"amount_mean_{window}"] = (
            grouped_amount.rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        )

    return features.replace([np.inf, -np.inf], np.nan)
