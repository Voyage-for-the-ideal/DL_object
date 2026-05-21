from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loader import CsvDataLoader


def test_loader_missing_partition_returns_empty(tmp_path: Path) -> None:
    (tmp_path / "daily").mkdir()
    loader = CsvDataLoader(tmp_path)
    assert loader.load_daily("20200101").empty


def test_loader_reads_basic_and_dates(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "market": ["主板"],
            "list_date": [19910403],
        }
    ).to_csv(tmp_path / "basic.csv", index=False)
    loader = CsvDataLoader(tmp_path)
    basic = loader.load_basic()
    assert basic.loc[0, "list_date"] == "19910403"
