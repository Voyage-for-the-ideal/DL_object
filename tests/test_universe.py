from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loader import CsvDataLoader
from src.data.universe import UniverseBuilder


def test_official_universe_excludes_bse_and_st(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "430001.BJ", "600001.SH"],
            "market": ["主板", "北交所", "主板"],
            "list_date": [20200101, 20200101, 20200101],
        }
    ).to_csv(tmp_path / "basic.csv", index=False)
    (tmp_path / "stock_st").mkdir()
    pd.DataFrame({"ts_code": ["600001.SH"], "trade_date": ["20200102"]}).to_csv(
        tmp_path / "stock_st" / "20200102.csv", index=False
    )
    universe = UniverseBuilder(CsvDataLoader(tmp_path)).get_official_universe("20200102")
    assert universe == {"000001.SZ"}


def test_hs300_weight_fallback_uses_past_file(tmp_path: Path) -> None:
    (tmp_path / "index_weight").mkdir()
    pd.DataFrame(
        {
            "index_code": ["000300.SH"],
            "con_code": ["000001.SZ"],
            "trade_date": ["20200131"],
            "weight": [1.0],
        }
    ).to_csv(tmp_path / "index_weight" / "202001_000300.SH.csv", index=False)
    pd.DataFrame(
        {
            "index_code": ["000300.SH"],
            "con_code": ["000002.SZ"],
            "trade_date": ["20200331"],
            "weight": [1.0],
        }
    ).to_csv(tmp_path / "index_weight" / "202003_000300.SH.csv", index=False)
    universe = UniverseBuilder(CsvDataLoader(tmp_path)).get_hs300_universe("20200215")
    assert universe == {"000001.SZ"}
