"""Simple daily backtest engine."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.backtest.execution import ExecutionConfig, ExecutionEngine
from src.backtest.metrics import summarize_nav
from src.backtest.portfolio import PortfolioState
from src.backtest.strategy import ScoreWeightedRiskControlStrategy, rebalance_orders
from src.data.calendar import TradingCalendar
from src.data.schema import TRADE_DATE, TS_CODE
from src.utils.io import ensure_dir


class BacktestEngine:
    def __init__(
        self,
        calendar: TradingCalendar,
        strategy: ScoreWeightedRiskControlStrategy | None = None,
        execution: ExecutionEngine | None = None,
        initial_cash: float = 1_000_000,
    ) -> None:
        self.calendar = calendar
        self.strategy = strategy or ScoreWeightedRiskControlStrategy()
        self.execution = execution or ExecutionEngine(ExecutionConfig())
        self.initial_cash = initial_cash

    def run(
        self,
        signals: pd.DataFrame,
        daily_panel: pd.DataFrame,
        start_date: str,
        end_date: str,
        output_dir: str | Path | None = None,
    ) -> dict[str, pd.DataFrame]:
        state = PortfolioState(cash=self.initial_cash)
        nav_rows: list[dict[str, float | str | int]] = []
        order_rows: list[dict[str, object]] = []
        trade_rows: list[dict[str, object]] = []
        position_rows: list[dict[str, object]] = []
        for trade_date in self.calendar.trade_dates_between(start_date, end_date):
            state.trade_date = trade_date
            daily = daily_panel[daily_panel[TRADE_DATE].astype(str) == trade_date].copy()
            prices = _prices_from_daily(daily)
            state.update_prices(prices)
            try:
                signal_date = self.calendar.previous_trade_date(trade_date)
            except ValueError:
                signal_date = trade_date
            today_signals = signals[
                signals.get("signal_date", signals.get(TRADE_DATE)).astype(str) == signal_date
            ]
            tradeable = set(daily[TS_CODE].astype(str))
            target_weights = self.strategy.target_weights(today_signals, tradeable=tradeable)
            orders = rebalance_orders(trade_date, state, target_weights, prices)
            for order in orders:
                trade, status = self.execution.execute_order(state, order, daily)
                order_rows.append({**order.__dict__, "status": status})
                if trade is not None:
                    trade_rows.append(trade.__dict__)
            nav_rows.append(
                {
                    TRADE_DATE: trade_date,
                    "cash": state.cash,
                    "total_value": state.total_value,
                    "nav": state.total_value / self.initial_cash,
                    "holding_count": state.holding_count(),
                }
            )
            for position in state.positions.values():
                position_rows.append({TRADE_DATE: trade_date, **position.__dict__})
        outputs = {
            "nav": pd.DataFrame(nav_rows),
            "orders": pd.DataFrame(order_rows),
            "trades": pd.DataFrame(trade_rows),
            "positions": pd.DataFrame(position_rows),
        }
        if output_dir is not None:
            self.save_outputs(outputs, output_dir)
        return outputs

    def save_outputs(self, outputs: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
        target = ensure_dir(output_dir)
        for name, frame in outputs.items():
            frame.to_csv(target / f"{name}.csv", index=False)
        metrics = summarize_nav(outputs["nav"])
        with (target / "backtest_metrics.json").open("w", encoding="utf-8") as file:
            json.dump(metrics, file, ensure_ascii=False, indent=2)
        if not outputs["nav"].empty:
            outputs["nav"].plot(x=TRADE_DATE, y="nav")
            plt.tight_layout()
            plt.savefig(target / "nav_curve.png")
            plt.close()


def _prices_from_daily(daily: pd.DataFrame) -> dict[str, float]:
    if daily.empty:
        return {}
    return {
        str(row[TS_CODE]): float(row.get("open", row.get("close", 0)))
        for _, row in daily.iterrows()
    }
