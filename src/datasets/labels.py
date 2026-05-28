"""Forward return label construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schema import TRADE_DATE, TS_CODE, normalize_trade_date_column


def add_forward_return_labels(
    price: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 3, 5),
    drop_missing: bool = False,
    *,
    market_df: pd.DataFrame | None = None,
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
    extra_5d_cols = []
    if 5 in horizons:
        ret = frame.groupby(TS_CODE, group_keys=False)["close"].pct_change()
        vol_20d = ret.groupby(frame[TS_CODE]).transform(
            lambda x: x.rolling(20, min_periods=20).std()
        ).clip(lower=1e-6)
        frame["label_5d_vol_norm"] = frame["label_5d"] / vol_20d
        frame["label_5d_cs_rank"] = frame.groupby(TRADE_DATE)["label_5d"].rank(pct=True)
        extra_5d_cols = ["label_5d_vol_norm", "label_5d_cs_rank"]
        if market_df is not None and not market_df.empty:
            mkt = market_df.copy()
            mkt = normalize_trade_date_column(mkt)
            mkt["mkt_close"] = pd.to_numeric(mkt["close"], errors="coerce")
            mkt = mkt.sort_values(TRADE_DATE)
            mkt["mkt_ret_5d"] = mkt["mkt_close"].pct_change(5).shift(-5)
            mkt_map = mkt.set_index(TRADE_DATE)["mkt_ret_5d"].to_dict()
            frame["label_5d_excess"] = frame["label_5d"].values - frame[TRADE_DATE].map(mkt_map).values
            frame["label_5d_excess"] = frame["label_5d_excess"].astype(float)
            extra_5d_cols.append("label_5d_excess")
    label_cols = [
        TRADE_DATE,
        TS_CODE,
        *[f"label_{h}d" for h in horizons],
        *extra_5d_cols,
    ]
    labels = frame[label_cols].replace(
        [np.inf, -np.inf], np.nan
    )
    if drop_missing:
        labels = labels.dropna(subset=[f"label_{h}d" for h in horizons], how="all")
    return labels.reset_index(drop=True)


def merge_labels(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    return features.merge(labels, on=[TRADE_DATE, TS_CODE], how="inner")
