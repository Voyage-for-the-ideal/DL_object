"""CSV data loader for the provided date-partitioned A-share data."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from src.data.schema import (
    INDEX_WEIGHT_COLUMNS,
    NEWS_COLUMNS,
    TRADE_DATE,
    TS_CODE,
    date_from_csv_path,
    ensure_columns,
    normalize_trade_date,
    normalize_trade_date_column,
)

LOGGER = logging.getLogger(__name__)


class CsvDataLoader:
    """Read raw CSV files without fitting statistics or constructing labels."""

    def __init__(self, data_root: str | Path) -> None:
        self.data_root = Path(data_root)

    def _read_csv(self, path: Path, required: list[str] | None = None) -> pd.DataFrame:
        if not path.exists():
            LOGGER.warning("CSV file missing: %s", path)
            return pd.DataFrame(columns=required or [])
        frame = pd.read_csv(path)
        if required:
            ensure_columns(frame, required, str(path))
        return frame

    def _partition_path(self, directory: str, trade_date: str) -> Path:
        return self.data_root / directory / f"{normalize_trade_date(trade_date)}.csv"

    def load_basic(self) -> pd.DataFrame:
        frame = self._read_csv(
            self.data_root / "basic.csv", required=[TS_CODE, "market", "list_date"]
        )
        if frame.empty:
            return frame
        frame = frame.copy()
        frame[TS_CODE] = frame[TS_CODE].astype(str)
        frame["list_date"] = frame["list_date"].map(normalize_trade_date)
        return frame

    def load_trade_calendar(self) -> pd.DataFrame:
        frame = self._read_csv(
            self.data_root / "trade_cal.csv",
            required=["exchange", "cal_date", "is_open", "pretrade_date"],
        )
        if frame.empty:
            return frame
        frame = frame.copy()
        frame["cal_date"] = frame["cal_date"].map(normalize_trade_date)
        frame["pretrade_date"] = frame["pretrade_date"].map(
            lambda value: normalize_trade_date(value) if not pd.isna(value) else ""
        )
        return frame.sort_values("cal_date").reset_index(drop=True)

    def load_daily(self, trade_date: str) -> pd.DataFrame:
        frame = self._read_csv(self._partition_path("daily", trade_date))
        return normalize_trade_date_column(frame)

    def load_metric(self, trade_date: str) -> pd.DataFrame:
        frame = self._read_csv(self._partition_path("metric", trade_date))
        return normalize_trade_date_column(frame)

    def load_moneyflow(self, trade_date: str) -> pd.DataFrame:
        frame = self._read_csv(self._partition_path("moneyflow", trade_date))
        return normalize_trade_date_column(frame)

    def load_st(self, trade_date: str) -> pd.DataFrame:
        frame = self._read_csv(self._partition_path("stock_st", trade_date))
        return normalize_trade_date_column(frame)

    def available_partition_dates(self, directory: str) -> list[str]:
        folder = self.data_root / directory
        if not folder.exists():
            return []
        return sorted(date_from_csv_path(path) for path in folder.glob("*.csv"))

    def latest_partition_date(self, directory: str, max_date: str | None = None) -> str | None:
        dates = self.available_partition_dates(directory)
        if max_date is not None:
            cutoff = normalize_trade_date(max_date)
            dates = [date for date in dates if date <= cutoff]
        return dates[-1] if dates else None

    def _index_weight_files(self, index_code: str | None = None) -> list[Path]:
        folder = self.data_root / "index_weight"
        if not folder.exists():
            return []
        files = sorted(folder.glob("*.csv"))
        if index_code is not None:
            files = [path for path in files if path.stem.endswith(f"_{index_code}")]
        return files

    def load_index_weight(
        self,
        trade_date: str | None = None,
        index_code: str = "000300.SH",
    ) -> pd.DataFrame:
        files = self._index_weight_files(index_code)
        if not files:
            return pd.DataFrame(columns=INDEX_WEIGHT_COLUMNS)
        if trade_date is not None:
            cutoff_month = normalize_trade_date(trade_date)[:6]
            files = [path for path in files if path.stem.split("_", 1)[0] <= cutoff_month]
            if not files:
                return pd.DataFrame(columns=INDEX_WEIGHT_COLUMNS)
            files = [files[-1]]

        frames = [self._read_csv(path, required=INDEX_WEIGHT_COLUMNS) for path in files]
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if frame.empty:
            return pd.DataFrame(columns=INDEX_WEIGHT_COLUMNS)
        frame = normalize_trade_date_column(frame)
        if trade_date is not None:
            frame = frame[frame[TRADE_DATE] <= normalize_trade_date(trade_date)]
            if not frame.empty:
                latest_date = frame[TRADE_DATE].max()
                frame = frame[frame[TRADE_DATE] == latest_date]
        return frame.reset_index(drop=True)

    def load_market(self, index_code: str) -> pd.DataFrame:
        frame = self._read_csv(self.data_root / "market" / f"{index_code}.csv")
        return normalize_trade_date_column(frame)

    def load_news(self, date: str) -> pd.DataFrame:
        frame = self._read_csv(self._partition_path("news", date), required=NEWS_COLUMNS)
        if frame.empty:
            return pd.DataFrame(columns=NEWS_COLUMNS)
        return frame

    def load_many_daily(
        self,
        trade_dates: list[str],
        show_progress: bool = False,
        desc: str = "load daily",
    ) -> pd.DataFrame:
        frames = [
            self.load_daily(date) for date in _progress(trade_dates, desc, show_progress)
        ]
        frames = [frame for frame in frames if not frame.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _progress(items: list[str], desc: str, show_progress: bool) -> Iterable[str]:
    if not show_progress:
        return items
    from tqdm.auto import tqdm

    return tqdm(items, desc=desc, unit="date")
