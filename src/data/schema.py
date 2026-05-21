"""Centralized schema names and date normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

TRADE_DATE = "trade_date"
TS_CODE = "ts_code"
SIGNAL_DATE = "signal_date"
NEXT_TRADE_DATE = "next_trade_date"

PRICE_COLUMNS = ["open", "high", "low", "close", "pre_close", "vol", "amount", "vwap"]
DAILY_REQUIRED_COLUMNS = [TS_CODE, TRADE_DATE, *PRICE_COLUMNS]
METRIC_COLUMNS = [
    "turnover_rate",
    "turnover_rate_f",
    "volume_ratio",
    "pe",
    "pe_ttm",
    "pb",
    "ps",
    "ps_ttm",
    "dv_ratio",
    "dv_ttm",
    "total_mv",
    "circ_mv",
]
NEWS_COLUMNS = ["datetime", "content", "title"]
INDEX_WEIGHT_COLUMNS = ["index_code", "con_code", TRADE_DATE, "weight"]


def normalize_trade_date(value: object) -> str:
    """Convert common date representations to ``YYYYMMDD``."""
    if pd.isna(value):
        raise ValueError("trade date cannot be null")
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y%m%d")
    text = str(value).strip()
    if not text:
        raise ValueError("trade date cannot be empty")
    if text.endswith(".0"):
        text = text[:-2]
    return pd.to_datetime(text, errors="raise").strftime("%Y%m%d") if "-" in text else text[:8]


def normalize_date_series(series: pd.Series) -> pd.Series:
    return series.map(normalize_trade_date).astype(str)


def normalize_trade_date_column(frame: pd.DataFrame, column: str = TRADE_DATE) -> pd.DataFrame:
    result = frame.copy()
    if column in result.columns:
        result[column] = normalize_date_series(result[column])
    return result


def ensure_columns(frame: pd.DataFrame, required: Iterable[str], source: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def date_from_csv_path(path: str | Path) -> str:
    return normalize_trade_date(Path(path).stem.split("_", 1)[0])
