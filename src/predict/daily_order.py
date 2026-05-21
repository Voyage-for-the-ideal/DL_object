"""Generate daily simulated-trading order suggestions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.backtest.portfolio import PortfolioState, Position
from src.backtest.strategy import ScoreWeightedRiskControlStrategy, rebalance_orders
from src.data.schema import NEXT_TRADE_DATE, SIGNAL_DATE, TS_CODE
from src.utils.io import ensure_dir

ORDER_COLUMNS = [
    SIGNAL_DATE,
    "trade_date",
    TS_CODE,
    "action",
    "target_weight",
    "target_shares",
    "estimated_price",
    "estimated_amount",
    "reason",
]


def load_portfolio(positions_file: str | Path | None, cash: float) -> PortfolioState:
    state = PortfolioState(cash=float(cash))
    if positions_file is None:
        return state
    frame = pd.read_csv(positions_file)
    for _, row in frame.iterrows():
        state.positions[str(row[TS_CODE])] = Position(
            ts_code=str(row[TS_CODE]),
            shares=int(row["shares"]),
            cost_price=float(row.get("cost_price", row.get("last_price", 0))),
            last_price=float(row.get("last_price", row.get("cost_price", 0))),
            buy_date=str(row.get("buy_date", "")),
        )
    return state


def generate_daily_orders(
    signal: pd.DataFrame,
    quotes: pd.DataFrame,
    state: PortfolioState,
    top_k: int = 20,
    max_single_weight: float = 0.10,
) -> pd.DataFrame:
    strategy = ScoreWeightedRiskControlStrategy(top_k=top_k, max_single_weight=max_single_weight)
    prices = {
        str(row[TS_CODE]): float(row.get("open", row.get("close", 0)))
        for _, row in quotes.iterrows()
    }
    tradeable = set(quotes[TS_CODE].astype(str))
    target_weights = strategy.target_weights(signal, tradeable=tradeable)
    trade_date = (
        str(signal[NEXT_TRADE_DATE].iloc[0])
        if NEXT_TRADE_DATE in signal.columns and not signal.empty
        else ""
    )
    signal_date = (
        str(signal[SIGNAL_DATE].iloc[0])
        if SIGNAL_DATE in signal.columns and not signal.empty
        else ""
    )
    orders = rebalance_orders(trade_date, state, target_weights, prices)
    rows: list[dict[str, object]] = []
    for order in orders:
        price = prices.get(order.ts_code, 0.0)
        rows.append(
            {
                SIGNAL_DATE: signal_date,
                "trade_date": trade_date,
                TS_CODE: order.ts_code,
                "action": order.action,
                "target_weight": target_weights.get(order.ts_code, 0.0),
                "target_shares": order.shares,
                "estimated_price": price,
                "estimated_amount": price * order.shares,
                "reason": order.reason,
            }
        )
    return pd.DataFrame(rows, columns=ORDER_COLUMNS)


def save_orders(orders: pd.DataFrame, output_root: str | Path, signal_date: str) -> Path:
    target = ensure_dir(Path(output_root) / "orders")
    path = target / f"{signal_date}_orders.csv"
    orders.to_csv(path, index=False)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal-file", required=True)
    parser.add_argument("--quotes-file", required=True)
    parser.add_argument("--positions-file", default=None)
    parser.add_argument("--cash", type=float, default=1_000_000)
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args(argv)
    signal = pd.read_csv(args.signal_file)
    quotes = pd.read_csv(args.quotes_file)
    state = load_portfolio(args.positions_file, args.cash)
    orders = generate_daily_orders(signal, quotes, state)
    signal_date = str(signal[SIGNAL_DATE].iloc[0]) if not signal.empty else "unknown"
    save_orders(orders, args.output_root, signal_date)


if __name__ == "__main__":
    main()
