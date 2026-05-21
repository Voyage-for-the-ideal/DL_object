from __future__ import annotations

import pandas as pd
import pytest

from src.datasets.labels import add_forward_return_labels


def test_label_1d_uses_t_plus_1_and_t_plus_2() -> None:
    price = pd.DataFrame(
        {
            "ts_code": ["A"] * 4,
            "trade_date": ["20200101", "20200102", "20200103", "20200104"],
            "close": [10.0, 20.0, 30.0, 60.0],
        }
    )
    labels = add_forward_return_labels(price, horizons=(1, 3, 5))
    assert labels.loc[0, "label_1d"] == pytest.approx(30 / 20 - 1)
    assert pd.isna(labels.loc[2, "label_1d"])
