from __future__ import annotations

import pandas as pd
import pytest

from src.backtest.metrics import benchmark_nav, max_drawdown, summarize_nav


def test_monotonic_nav_drawdown_zero() -> None:
    nav = pd.Series([1.0, 1.1, 1.2])
    assert max_drawdown(nav) == 0.0


def test_known_cumulative_return() -> None:
    metrics = summarize_nav(pd.DataFrame({"nav": [1.0, 1.1]}))
    assert metrics["cumulative_return"] == pytest.approx(0.1)


def test_benchmark_nav_normalizes_prices() -> None:
    market = pd.DataFrame({"trade_date": ["20200101", "20200102"], "close": [10.0, 11.0]})
    nav = benchmark_nav(market, "20200101", "20200102")
    assert nav.loc[1, "benchmark_nav"] == pytest.approx(1.1)
