from __future__ import annotations

import pandas as pd

from src.features.price_features import build_price_features


def _daily(close_t4: float = 40.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": ["A"] * 4,
            "trade_date": ["20200101", "20200102", "20200103", "20200104"],
            "open": [10, 20, 30, close_t4],
            "high": [11, 21, 31, close_t4],
            "low": [9, 19, 29, close_t4],
            "close": [10, 20, 30, close_t4],
            "pre_close": [9, 10, 20, 30],
            "vol": [100, 200, 300, 400],
            "amount": [1000, 2000, 3000, 4000],
            "vwap": [10, 20, 30, close_t4],
        }
    )


def test_price_features_do_not_use_future_rows() -> None:
    base = build_price_features(_daily())
    changed = build_price_features(_daily(999.0))
    row_base = base[base["trade_date"] == "20200103"].reset_index(drop=True)
    row_changed = changed[changed["trade_date"] == "20200103"].reset_index(drop=True)
    pd.testing.assert_series_equal(row_base.iloc[0], row_changed.iloc[0], check_names=False)
