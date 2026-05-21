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
