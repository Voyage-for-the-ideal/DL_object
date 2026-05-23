from __future__ import annotations

import pandas as pd

from src.datasets.split import TimeSplit, split_by_date
from src.datasets.tabular_dataset import TabularDataset, select_feature_columns
from src.datasets.window_dataset import WindowDataset


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["20200101", "20200102", "20200103", "20200104"],
            "ts_code": ["A", "A", "A", "A"],
            "feature": [1.0, 2.0, 3.0, 4.0],
            "close": [1.0, 2.0, 3.0, 4.0],
            "label_1d": [0.1, 0.2, 0.3, 0.4],
        }
    )


def test_feature_selection_excludes_labels_and_prices() -> None:
    assert select_feature_columns(_panel()) == ["feature"]


def test_tabular_dataset_shapes() -> None:
    dataset = TabularDataset.from_frame(_panel())
    assert dataset.X.shape == (4, 1)
    assert dataset.y.shape == (4,)


def test_window_dataset_ends_at_signal_date_without_future() -> None:
    dataset = WindowDataset.from_frame(_panel(), lookback=2)
    assert dataset.X.shape == (3, 2, 1)
    assert dataset.index.iloc[0]["trade_date"] == "20200102"


def test_window_dataset_filters_endpoint_dates_and_keeps_history() -> None:
    train = _panel().iloc[:2]
    valid = _panel().iloc[2:]
    history = pd.concat([train, valid], ignore_index=True)
    dataset = WindowDataset.from_frame(history, lookback=3, end_dates=valid["trade_date"])
    assert dataset.X.shape == (2, 3, 1)
    assert dataset.index["trade_date"].tolist() == ["20200103", "20200104"]
    assert dataset.X[0, :, 0].tolist() == [1.0, 2.0, 3.0]


def test_window_dataset_groups_each_stock_separately() -> None:
    panel = pd.concat(
        [
            _panel().assign(ts_code="A"),
            _panel().assign(ts_code="B", feature=[10.0, 20.0, 30.0, 40.0]),
        ],
        ignore_index=True,
    )
    dataset = WindowDataset.from_frame(panel, lookback=4, end_dates=["20200104"])
    assert dataset.X.shape == (2, 4, 1)
    assert dataset.index["ts_code"].tolist() == ["A", "B"]
    assert dataset.X[1, :, 0].tolist() == [10.0, 20.0, 30.0, 40.0]


def test_time_split_is_date_based() -> None:
    train, valid = split_by_date(
        _panel(), TimeSplit("20200101", "20200102", "20200103", "20200104")
    )
    assert set(train["trade_date"]) == {"20200101", "20200102"}
    assert set(valid["trade_date"]) == {"20200103", "20200104"}
