from __future__ import annotations

import pandas as pd

from src.backtest.execution import ExecutionConfig, ExecutionEngine, Order
from src.backtest.portfolio import PortfolioState, Position


def _daily(pct_chg: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": ["A"],
            "open": [10.0],
            "close": [10.0],
            "vol": [1000],
            "amount": [10000],
            "pct_chg": [pct_chg],
        }
    )


def test_t_plus_one_blocks_same_day_sell() -> None:
    state = PortfolioState(cash=0, positions={"A": Position("A", 100, 10, 10, "20200102")})
    engine = ExecutionEngine()
    trade, status = engine.execute_order(state, Order("20200102", "A", "sell", 100), _daily())
    assert trade is None
    assert status == "t_plus_one_blocked"


def test_buy_rounds_down_to_lot() -> None:
    state = PortfolioState(cash=2000)
    engine = ExecutionEngine(ExecutionConfig(slippage_rate=0.0))
    trade, status = engine.execute_order(state, Order("20200102", "A", "buy", 101), _daily())
    assert status == "filled"
    assert trade is not None and trade.shares == 100


def test_limit_up_buy_and_limit_down_sell_blocked() -> None:
    state = PortfolioState(cash=2000, positions={"A": Position("A", 100, 10, 10, "20200101")})
    engine = ExecutionEngine()
    assert (
        engine.execute_order(state, Order("20200102", "A", "buy", 100), _daily(10.0))[1]
        == "limit_up_buy_blocked"
    )
    assert (
        engine.execute_order(state, Order("20200102", "A", "sell", 100), _daily(-10.0))[1]
        == "limit_down_sell_blocked"
    )
