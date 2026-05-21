"""Order execution with A-share constraints."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.backtest.portfolio import PortfolioState, Position
from src.data.schema import TS_CODE


@dataclass
class ExecutionConfig:
    commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.001
    slippage_rate: float = 0.0005
    min_lot_size: int = 100
    execution_price: str = "open"
    block_limit_up_buy: bool = True
    block_limit_down_sell: bool = True
    enforce_t_plus_one: bool = True


@dataclass
class Order:
    trade_date: str
    ts_code: str
    action: str
    shares: int
    reason: str = ""
    target_weight: float = 0.0


@dataclass
class Trade:
    trade_date: str
    ts_code: str
    action: str
    shares: int
    price: float
    amount: float
    fee: float
    tax: float
    reason: str


class ExecutionEngine:
    def __init__(self, config: ExecutionConfig | None = None) -> None:
        self.config = config or ExecutionConfig()

    def quote_for(self, daily: pd.DataFrame, ts_code: str) -> pd.Series | None:
        rows = daily[daily[TS_CODE].astype(str) == ts_code]
        if rows.empty:
            return None
        return rows.iloc[0]

    def is_tradeable(self, quote: pd.Series | None) -> bool:
        if quote is None:
            return False
        vol = float(pd.to_numeric(quote.get("vol", 0), errors="coerce") or 0)
        amount = float(pd.to_numeric(quote.get("amount", 0), errors="coerce") or 0)
        return vol > 0 and amount >= 0

    def is_limit_up(self, quote: pd.Series) -> bool:
        pct_chg = pd.to_numeric(quote.get("pct_chg", 0), errors="coerce")
        return bool(pd.notna(pct_chg) and float(pct_chg) >= 9.8)

    def is_limit_down(self, quote: pd.Series) -> bool:
        pct_chg = pd.to_numeric(quote.get("pct_chg", 0), errors="coerce")
        return bool(pd.notna(pct_chg) and float(pct_chg) <= -9.8)

    def execution_price(self, quote: pd.Series, action: str) -> float:
        base = float(
            pd.to_numeric(
                quote.get(self.config.execution_price, quote.get("open")), errors="coerce"
            )
        )
        slip = self.config.slippage_rate
        return base * (1 + slip if action == "buy" else 1 - slip)

    def execute_order(
        self,
        state: PortfolioState,
        order: Order,
        daily: pd.DataFrame,
    ) -> tuple[Trade | None, str]:
        quote = self.quote_for(daily, order.ts_code)
        if not self.is_tradeable(quote):
            return None, "not_tradeable"
        assert quote is not None
        if order.action == "buy" and self.config.block_limit_up_buy and self.is_limit_up(quote):
            return None, "limit_up_buy_blocked"
        if (
            order.action == "sell"
            and self.config.block_limit_down_sell
            and self.is_limit_down(quote)
        ):
            return None, "limit_down_sell_blocked"
        if order.action == "sell":
            position = state.positions.get(order.ts_code)
            if position is None or position.shares <= 0:
                return None, "no_position"
            if self.config.enforce_t_plus_one and position.buy_date == order.trade_date:
                return None, "t_plus_one_blocked"
            shares = min(order.shares, position.shares)
            return self._sell(state, order, quote, shares), "filled"
        if order.action == "buy":
            shares = self._round_buy_shares(order.shares)
            return self._buy(state, order, quote, shares)
        return None, "unknown_action"

    def _round_buy_shares(self, shares: int) -> int:
        lot = self.config.min_lot_size
        return max((int(shares) // lot) * lot, 0)

    def _buy(
        self,
        state: PortfolioState,
        order: Order,
        quote: pd.Series,
        shares: int,
    ) -> tuple[Trade | None, str]:
        if shares <= 0:
            return None, "below_min_lot"
        price = self.execution_price(quote, "buy")
        gross = shares * price
        fee = gross * self.config.commission_rate
        total_cost = gross + fee
        while shares > 0 and total_cost > state.cash:
            shares -= self.config.min_lot_size
            gross = shares * price
            fee = gross * self.config.commission_rate
            total_cost = gross + fee
        if shares <= 0:
            return None, "insufficient_cash"
        state.cash -= total_cost
        existing = state.positions.get(order.ts_code)
        if existing is None:
            state.positions[order.ts_code] = Position(
                order.ts_code, shares, price, price, order.trade_date
            )
        else:
            total_shares = existing.shares + shares
            existing.cost_price = (existing.cost_price * existing.shares + gross) / total_shares
            existing.shares = total_shares
            existing.last_price = price
        trade = Trade(
            order.trade_date, order.ts_code, "buy", shares, price, gross, fee, 0.0, order.reason
        )
        return trade, "filled"

    def _sell(
        self,
        state: PortfolioState,
        order: Order,
        quote: pd.Series,
        shares: int,
    ) -> Trade | None:
        if shares <= 0:
            return None
        position = state.positions[order.ts_code]
        price = self.execution_price(quote, "sell")
        gross = shares * price
        fee = gross * self.config.commission_rate
        tax = gross * self.config.stamp_tax_rate
        state.cash += gross - fee - tax
        position.shares -= shares
        position.last_price = price
        if position.shares <= 0:
            del state.positions[order.ts_code]
        return Trade(
            order.trade_date, order.ts_code, "sell", shares, price, gross, fee, tax, order.reason
        )
