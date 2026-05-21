"""Strategies converting scores into target weights and orders."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.execution import Order
from src.backtest.portfolio import PortfolioState
from src.data.schema import TS_CODE


class TopKEqualWeightStrategy:
    def __init__(self, top_k: int = 20) -> None:
        self.top_k = top_k

    def target_weights(
        self, signals: pd.DataFrame, tradeable: set[str] | None = None
    ) -> dict[str, float]:
        frame = (
            _filter_signals(signals, tradeable)
            .sort_values("score", ascending=False)
            .head(self.top_k)
        )
        if frame.empty:
            return {}
        weight = 1.0 / len(frame)
        return {str(ts_code): weight for ts_code in frame[TS_CODE]}


class ScoreWeightedRiskControlStrategy:
    def __init__(
        self,
        top_k: int = 20,
        max_single_weight: float = 0.10,
        max_industry_weight: float = 0.30,
    ) -> None:
        self.top_k = top_k
        self.max_single_weight = max_single_weight
        self.max_industry_weight = max_industry_weight

    def target_weights(
        self, signals: pd.DataFrame, tradeable: set[str] | None = None
    ) -> dict[str, float]:
        frame = (
            _filter_signals(signals, tradeable)
            .sort_values("score", ascending=False)
            .head(self.top_k)
        )
        if frame.empty:
            return {}
        ranks = np.arange(len(frame), 0, -1, dtype=float)
        weights = ranks / ranks.sum()
        weights = np.minimum(weights, self.max_single_weight)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        weights = np.minimum(weights, self.max_single_weight)
        leftover = max(0.0, 1.0 - float(weights.sum()))
        if leftover > 0 and len(weights) > 0:
            weights = weights + leftover / len(weights)
            weights = np.minimum(weights, self.max_single_weight)
        return {str(ts_code): float(weight) for ts_code, weight in zip(frame[TS_CODE], weights)}


def _filter_signals(signals: pd.DataFrame, tradeable: set[str] | None) -> pd.DataFrame:
    frame = signals.copy()
    if tradeable is not None:
        frame = frame[frame[TS_CODE].astype(str).isin(tradeable)]
    return frame.dropna(subset=["score"])


def rebalance_orders(
    trade_date: str,
    state: PortfolioState,
    target_weights: dict[str, float],
    prices: dict[str, float],
) -> list[Order]:
    total_value = state.total_value
    orders: list[Order] = []
    for ts_code, position in list(state.positions.items()):
        target_value = target_weights.get(ts_code, 0.0) * total_value
        current_value = position.shares * prices.get(ts_code, position.last_price)
        if current_value > target_value:
            shares = int((current_value - target_value) / prices.get(ts_code, position.last_price))
            if shares > 0:
                orders.append(Order(trade_date, ts_code, "sell", shares, "rebalance_sell"))
    for ts_code, weight in target_weights.items():
        price = prices.get(ts_code)
        if price is None or price <= 0:
            continue
        current_position = state.positions.get(ts_code)
        current_shares = current_position.shares if current_position is not None else 0
        target_shares = int((total_value * weight) / price)
        if target_shares > current_shares:
            orders.append(
                Order(
                    trade_date,
                    ts_code,
                    "buy",
                    target_shares - current_shares,
                    "rebalance_buy",
                    weight,
                )
            )
    return orders
