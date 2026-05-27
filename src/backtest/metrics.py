"""Backtest metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schema import TRADE_DATE


def max_drawdown(nav: pd.Series) -> float:
    if nav.empty:
        return float("nan")
    drawdown = nav / nav.cummax() - 1
    return float(abs(drawdown.min()))


def summarize_nav(nav_frame: pd.DataFrame, periods_per_year: int = 252) -> dict[str, float]:
    nav = pd.to_numeric(nav_frame["nav"], errors="coerce").dropna()
    returns = nav.pct_change().dropna()
    cumulative = float(nav.iloc[-1] / nav.iloc[0] - 1) if len(nav) >= 2 else 0.0
    annualized = float((1 + cumulative) ** (periods_per_year / max(len(nav) - 1, 1)) - 1)
    volatility = float(returns.std(ddof=1) * np.sqrt(periods_per_year)) if len(returns) > 1 else 0.0
    sharpe = (
        float(returns.mean() / returns.std(ddof=1) * np.sqrt(periods_per_year))
        if returns.std(ddof=1) > 0
        else 0.0
    )
    return {
        "cumulative_return": cumulative,
        "annualized_return": annualized,
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(nav),
    }


def compare_nav(
    strategy_nav: pd.DataFrame,
    benchmark_nav: pd.DataFrame | None = None,
    benchmark_label: str = "benchmark",
    periods_per_year: int = 252,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {
        "strategy": summarize_nav(strategy_nav, periods_per_year=periods_per_year),
    }
    if benchmark_nav is not None and not benchmark_nav.empty:
        bench_col = (
            "benchmark_nav" if "benchmark_nav" in benchmark_nav.columns
            else benchmark_nav.columns[-1]
        )
        result[f"benchmark_{benchmark_label}"] = summarize_nav(
            benchmark_nav.rename(columns={bench_col: "nav"}),
            periods_per_year=periods_per_year,
        )
    return result


def benchmark_nav(
    market: pd.DataFrame,
    start_date: str,
    end_date: str,
    price_column: str = "close",
) -> pd.DataFrame:
    """Convert benchmark market prices, e.g. HS300 or SSE, into normalized NAV."""
    from src.data.schema import normalize_trade_date

    start = normalize_trade_date(start_date)
    end = normalize_trade_date(end_date)
    frame = market.copy()
    frame[TRADE_DATE] = frame[TRADE_DATE].astype(str)
    frame = frame[(frame[TRADE_DATE] >= start) & (frame[TRADE_DATE] <= end)]
    frame = frame.sort_values(TRADE_DATE)
    if frame.empty:
        return pd.DataFrame(columns=[TRADE_DATE, "benchmark_nav"])
    prices = pd.to_numeric(frame[price_column], errors="coerce")
    frame["benchmark_nav"] = prices / prices.iloc[0]
    return frame[[TRADE_DATE, "benchmark_nav"]].reset_index(drop=True)
