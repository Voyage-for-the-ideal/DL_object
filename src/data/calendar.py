"""Trading calendar utilities."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass

import pandas as pd

from src.data.schema import normalize_trade_date


@dataclass(frozen=True)
class TradingCalendar:
    """A sorted list of open trading days."""

    trade_dates: list[str]

    @classmethod
    def from_frame(cls, calendar: pd.DataFrame) -> "TradingCalendar":
        date_col = "cal_date" if "cal_date" in calendar.columns else "trade_date"
        frame = calendar.copy()
        if "is_open" in frame.columns:
            frame = frame[frame["is_open"].astype(int) == 1]
        dates = sorted({normalize_trade_date(value) for value in frame[date_col]})
        return cls(dates)

    def is_trading_day(self, date: str) -> bool:
        return normalize_trade_date(date) in set(self.trade_dates)

    def previous_trade_date(self, date: str) -> str:
        target = normalize_trade_date(date)
        index = bisect_left(self.trade_dates, target) - 1
        if index < 0:
            raise ValueError(f"No previous trading day before {target}")
        return self.trade_dates[index]

    def next_trade_date(self, date: str, n: int = 1) -> str:
        if n < 1:
            raise ValueError("n must be positive")
        target = normalize_trade_date(date)
        index = bisect_right(self.trade_dates, target) + n - 1
        if index >= len(self.trade_dates):
            raise ValueError(f"No T+{n} trading day after {target}")
        return self.trade_dates[index]

    def offset(self, date: str, n: int) -> str:
        target = normalize_trade_date(date)
        if target not in self.trade_dates:
            raise ValueError(f"{target} is not a trading day")
        index = self.trade_dates.index(target) + n
        if index < 0 or index >= len(self.trade_dates):
            raise ValueError(f"Offset {n} from {target} is outside calendar")
        return self.trade_dates[index]

    def trade_dates_between(self, start: str, end: str) -> list[str]:
        left = bisect_left(self.trade_dates, normalize_trade_date(start))
        right = bisect_right(self.trade_dates, normalize_trade_date(end))
        return self.trade_dates[left:right]
