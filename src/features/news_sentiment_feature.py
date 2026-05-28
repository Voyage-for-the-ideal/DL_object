"""Per-stock per-day news sentiment feature generator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.schema import TRADE_DATE, TS_CODE


class NewsSentimentFeatureGenerator:
    """Generate per-stock per-day sentiment features from news data."""

    def __init__(
        self,
        sentiment_model: Any,  # FinBERTSentimentModel
        device: str = "cpu",
    ):
        self.model = sentiment_model
        self.device = device

    def generate(
        self,
        stock_news: pd.DataFrame,
    ) -> pd.DataFrame:
        """Aggregate news sentiment per stock per day.

        Args:
            stock_news: DataFrame with columns [trade_date, ts_code, content, title].

        Returns:
            DataFrame with sentiment features per stock-day.
        """
        if stock_news.empty:
            return pd.DataFrame(columns=[
                TRADE_DATE, TS_CODE, "sentiment_score", "sentiment_intensity",
                "sentiment_news_count", "sentiment_latest_intensity",
            ])

        # Prepare texts
        texts = []
        for _, row in stock_news.iterrows():
            title = str(row.get("title", ""))
            content = str(row.get("content", ""))
            text = title + " " + content[:500]
            texts.append(text)

        predictions = self.model.predict(texts, device=self.device)
        # predictions: [N, 6] = [class, prob_pos, prob_neu, prob_neg, intensity]

        result = stock_news[[TRADE_DATE, TS_CODE]].copy()
        result["_class"] = predictions[:, 0]
        result["_prob_pos"] = predictions[:, 1]
        result["_prob_neg"] = predictions[:, 3]
        result["_intensity"] = predictions[:, 4]

        # Per-stock per-day aggregation
        grouped = result.groupby([TRADE_DATE, TS_CODE], as_index=False).agg(
            sentiment_score=("_intensity", "mean"),
            sentiment_intensity=("_intensity", lambda x: x.abs().mean()),
            sentiment_news_count=("_intensity", "count"),
            sentiment_latest_intensity=("_intensity", "last"),
            sentiment_pos_ratio=("_prob_pos", "mean"),
            sentiment_neg_ratio=("_prob_neg", "mean"),
        )
        return grouped

    def save(self, path: str | Path) -> None:
        self.model.save(path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "NewsSentimentFeatureGenerator":
        from src.models.news_sentiment import FinBERTSentimentModel
        model = FinBERTSentimentModel.load(path)
        return cls(sentiment_model=model, device=device)
