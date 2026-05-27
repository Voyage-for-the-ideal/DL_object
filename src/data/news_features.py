"""Market-level news aggregation helpers."""

from __future__ import annotations

import pandas as pd

from src.data.schema import normalize_trade_date


def aggregate_news(news: pd.DataFrame, date: str | None = None) -> pd.DataFrame:
    """Aggregate raw news rows into one market-level text row per day."""
    columns = ["trade_date", "news_count", "title_count", "avg_content_length", "aggregated_text"]
    if news.empty:
        if date is None:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(
            [
                {
                    "trade_date": normalize_trade_date(date),
                    "news_count": 0,
                    "title_count": 0,
                    "avg_content_length": 0.0,
                    "aggregated_text": "",
                }
            ]
        )
    frame = news.copy()
    frame["trade_date"] = pd.to_datetime(frame["datetime"]).dt.strftime("%Y%m%d")
    frame["content"] = frame.get("content", "").fillna("").astype(str)
    frame["title"] = frame.get("title", "").fillna("").astype(str)
    grouped = frame.groupby("trade_date", sort=True).agg(
        news_count=("content", "size"),
        title_count=("title", lambda value: int((value.astype(str).str.len() > 0).sum())),
        avg_content_length=("content", lambda value: float(value.astype(str).str.len().mean())),
        aggregated_text=("content", lambda value: " ".join(value.astype(str))),
    )
    grouped["aggregated_text"] = (
        grouped["aggregated_text"]
        + " "
        + frame.groupby("trade_date")["title"].agg(lambda value: " ".join(value.astype(str)))
    )
    return grouped.reset_index()[columns]


def aggregate_news_titles(news: pd.DataFrame, date: str | None = None) -> pd.DataFrame:
    """Aggregate raw news into one market-level row per day using titles only.

    Uses only the 'title' field for aggregated_text to avoid exceeding
    BERT's 512-token limit when hundreds of news items are concatenated.
    """
    columns = ["trade_date", "news_count", "title_count", "avg_content_length", "aggregated_text"]
    if news.empty:
        if date is None:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(
            [
                {
                    "trade_date": normalize_trade_date(date),
                    "news_count": 0,
                    "title_count": 0,
                    "avg_content_length": 0.0,
                    "aggregated_text": "",
                }
            ]
        )
    frame = news.copy()
    frame["trade_date"] = pd.to_datetime(frame["datetime"]).dt.strftime("%Y%m%d")
    frame["title"] = frame.get("title", "").fillna("").astype(str)
    grouped = frame.groupby("trade_date", sort=True).agg(
        news_count=("title", "size"),
        title_count=("title", lambda v: int((v.astype(str).str.len() > 0).sum())),
        avg_content_length=("title", lambda v: float(v.astype(str).str.len().mean())),
        aggregated_text=("title", lambda v: " ".join(v.astype(str))),
    )
    return grouped.reset_index()[columns]


def aggregate_stock_news(news: pd.DataFrame) -> pd.DataFrame:
    """Aggregate news per (trade_date, ts_code) for stock-level features.

    Args:
        news: Raw news DataFrame with ts_code column (from extract_stock_codes).
              Only rows where ts_code != "" are included.

    Returns:
        DataFrame with columns:
        [trade_date, ts_code, news_count, title_count, avg_content_length, aggregated_text]
        One row per unique (trade_date, ts_code) combination.
    """
    columns = [
        "trade_date", "ts_code", "news_count", "title_count",
        "avg_content_length", "aggregated_text",
    ]
    if news.empty:
        return pd.DataFrame(columns=columns)

    frame = news.copy()
    frame["trade_date"] = pd.to_datetime(frame["datetime"]).dt.strftime("%Y%m%d")

    has_ts = "ts_code" in frame.columns
    if not has_ts:
        return pd.DataFrame(columns=columns)

    frame = frame[frame["ts_code"].astype(str).str.strip() != ""]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    frame["content"] = frame.get("content", "").fillna("").astype(str)
    frame["title"] = frame.get("title", "").fillna("").astype(str)

    grouped = frame.groupby(["trade_date", "ts_code"], sort=True).agg(
        news_count=("content", "size"),
        title_count=("title", lambda v: int((v.astype(str).str.len() > 0).sum())),
        avg_content_length=("content", lambda v: float(v.astype(str).str.len().mean())),
        aggregated_text=("content", lambda v: " ".join(v.astype(str))),
    )
    title_texts = frame.groupby(["trade_date", "ts_code"])["title"].agg(
        lambda v: " ".join(v.astype(str))
    )
    grouped["aggregated_text"] = grouped["aggregated_text"] + " " + title_texts

    return grouped.reset_index()[columns]
