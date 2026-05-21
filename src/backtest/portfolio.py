"""Portfolio state structures."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Position:
    ts_code: str
    shares: int
    cost_price: float
    last_price: float
    buy_date: str

    @property
    def market_value(self) -> float:
        return float(self.shares * self.last_price)


@dataclass
class PortfolioState:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    trade_date: str = ""

    @property
    def total_value(self) -> float:
        return float(self.cash + sum(position.market_value for position in self.positions.values()))

    def update_prices(self, prices: dict[str, float]) -> None:
        for ts_code, price in prices.items():
            if ts_code in self.positions:
                self.positions[ts_code].last_price = float(price)

    def holding_count(self) -> int:
        return sum(1 for position in self.positions.values() if position.shares > 0)
