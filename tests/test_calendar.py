from __future__ import annotations

import pandas as pd
import pytest

from src.data.calendar import TradingCalendar


def test_trading_calendar_offsets() -> None:
    calendar = TradingCalendar.from_frame(
        pd.DataFrame({"cal_date": ["20200101", "20200102", "20200103"], "is_open": [0, 1, 1]})
    )
    assert not calendar.is_trading_day("20200101")
    assert calendar.next_trade_date("20200102") == "20200103"
    assert calendar.previous_trade_date("20200103") == "20200102"
    assert calendar.trade_dates_between("20200101", "20200103") == ["20200102", "20200103"]
    with pytest.raises(ValueError):
        calendar.next_trade_date("20200103")
