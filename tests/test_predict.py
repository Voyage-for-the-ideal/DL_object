from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.backtest.portfolio import PortfolioState
from src.data.calendar import TradingCalendar
from src.data.loader import CsvDataLoader
from src.features.pipeline import build_latest_feature_frame
from src.features.preprocess import TrainOnlyPreprocessor
from src.models.linear import SklearnRegressorModel
from src.predict.daily_order import ORDER_COLUMNS, generate_daily_orders
from src.predict.daily_signal import generate_daily_signal, latest_available_signal_date


def test_latest_signal_date_ignores_future_missing_files(tmp_path: Path) -> None:
    (tmp_path / "daily").mkdir()
    pd.DataFrame({"ts_code": ["A"], "trade_date": ["20200102"]}).to_csv(
        tmp_path / "daily" / "20200102.csv", index=False
    )
    assert latest_available_signal_date(CsvDataLoader(tmp_path), max_date="20200103") == "20200102"


def test_latest_feature_frame_does_not_need_future_daily(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "ts_code": ["A"],
            "market": ["主板"],
            "list_date": [20190101],
        }
    ).to_csv(tmp_path / "basic.csv", index=False)
    (tmp_path / "daily").mkdir()
    for date, close in [("20200101", 10.0), ("20200102", 11.0)]:
        pd.DataFrame(
            {
                "ts_code": ["A"],
                "trade_date": [date],
                "open": [close],
                "high": [close],
                "low": [close],
                "close": [close],
                "pre_close": [close - 1],
                "vol": [100],
                "amount": [1000],
                "vwap": [close],
            }
        ).to_csv(tmp_path / "daily" / f"{date}.csv", index=False)
    loader = CsvDataLoader(tmp_path)
    calendar = TradingCalendar(["20200101", "20200102", "20200103"])
    latest = build_latest_feature_frame(
        loader, calendar, "20200102", lookback=2, universe_mode="official"
    )
    assert set(latest["trade_date"]) == {"20200102"}


def test_signal_and_order_fields_complete() -> None:
    features = pd.DataFrame(
        {
            "trade_date": ["20200101", "20200101", "20200101"],
            "ts_code": ["A", "B", "C"],
            "feature": [1.0, 2.0, 3.0],
            "label_1d": [0.1, 0.2, 0.3],
        }
    )
    pre = TrainOnlyPreprocessor().fit(features, ["feature"])
    model = SklearnRegressorModel.ridge().fit(
        pre.transform(features)[["feature"]].to_numpy(), features["label_1d"].to_numpy()
    )
    signal = generate_daily_signal(features, model, pre, "20200101", "20200102")
    assert list(signal.columns) == [
        "signal_date",
        "next_trade_date",
        "ts_code",
        "score",
        "rank",
        "model_name",
    ]
    quotes = pd.DataFrame(
        {
            "ts_code": ["A", "B", "C"],
            "open": [10.0, 10.0, 10.0],
            "close": [10.0, 10.0, 10.0],
        }
    )
    orders = generate_daily_orders(signal, quotes, PortfolioState(cash=10000), top_k=2)
    assert list(orders.columns) == ORDER_COLUMNS
