"""Money-flow feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schema import TRADE_DATE, TS_CODE, normalize_trade_date_column

LEVELS = ("sm", "md", "lg", "elg")


def build_moneyflow_features(
    moneyflow: pd.DataFrame,
    windows: tuple[int, ...] = (5, 10, 20),
) -> pd.DataFrame:
    if moneyflow.empty:
        return pd.DataFrame(columns=[TRADE_DATE, TS_CODE])
    frame = normalize_trade_date_column(moneyflow).copy()
    frame = frame.sort_values([TS_CODE, TRADE_DATE]).reset_index(drop=True)
    result = frame[[TRADE_DATE, TS_CODE]].copy()
    for column in frame.columns:
        if column not in {TRADE_DATE, TS_CODE}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    for level in LEVELS:
        buy_amount = frame.get(f"buy_{level}_amount", 0)
        sell_amount = frame.get(f"sell_{level}_amount", 0)
        buy_vol = frame.get(f"buy_{level}_vol", 0)
        sell_vol = frame.get(f"sell_{level}_vol", 0)
        result[f"mf_{level}_amount_diff"] = buy_amount - sell_amount
        result[f"mf_{level}_vol_diff"] = buy_vol - sell_vol

    result["mf_net_amount"] = pd.to_numeric(frame.get("net_mf_amount", 0), errors="coerce")
    result["mf_net_vol"] = pd.to_numeric(frame.get("net_mf_vol", 0), errors="coerce")
    denominator = sum(
        pd.to_numeric(frame.get(f"buy_{level}_amount", 0), errors="coerce") for level in LEVELS
    )
    denominator = denominator.replace(0, np.nan)
    result["mf_net_amount_ratio"] = result["mf_net_amount"] / denominator

    grouped = result.groupby(frame[TS_CODE], group_keys=False)
    for window in windows:
        result[f"mf_net_amount_mean_{window}"] = (
            grouped["mf_net_amount"]
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        result[f"mf_net_ratio_mean_{window}"] = (
            grouped["mf_net_amount_ratio"]
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

    result["mf_net_amount_rank"] = result.groupby(TRADE_DATE)["mf_net_amount"].rank(pct=True)
    return result.replace([np.inf, -np.inf], np.nan)
