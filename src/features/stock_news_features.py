"""Stock-level TF-IDF + SVD news features."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.schema import STOCK_NEWS_STAT_COLUMNS, STOCK_NEWS_TFIDF_PREFIX, TRADE_DATE, TS_CODE


class StockNewsFeatureGenerator:
    """Fit-on-train-only: TF-IDF → TruncatedSVD for per-stock-per-day news.

    Pipeline:
    1. TF-IDF vectorizer fits on training-period stock news text only.
    2. TruncatedSVD reduces TF-IDF to svd_dim dimensions.
    3. Basic stats (news_count, title_count, avg_content_length) per stock per day.

    Column names use the stock_news_ prefix to avoid collision with market-level
    news_count/title_count columns from NewsFinbertFeatureGenerator.
    """

    def __init__(
        self,
        tfidf_max_features: int = 256,
        svd_dim: int = 16,
    ) -> None:
        self.tfidf_max_features = tfidf_max_features
        self.svd_dim = svd_dim
        self.vectorizer = TfidfVectorizer(
            max_features=tfidf_max_features,
            sublinear_tf=True,  # 1 + log(tf)
        )
        self.svd = TruncatedSVD(n_components=svd_dim, random_state=42)
        self.fitted = False

    @property
    def feature_names(self) -> list[str]:
        return [f"{STOCK_NEWS_TFIDF_PREFIX}{i:02d}" for i in range(self.svd_dim)]

    def fit(self, aggregated_stock_news: pd.DataFrame) -> "StockNewsFeatureGenerator":
        """Fit TF-IDF vocabulary and SVD on training-period stock news only."""
        texts = (
            aggregated_stock_news.get("aggregated_text", pd.Series(dtype=str))
            .fillna("")
            .astype(str)
        )
        if texts.str.len().sum() == 0:
            self.vectorizer.fit(["empty"])
        else:
            self.vectorizer.fit(texts)
        tfidf_matrix = self.vectorizer.transform(texts)
        dense = tfidf_matrix.toarray()
        n_samples = dense.shape[0]
        # Pad to ensure both dimensions >= svd_dim for SVD
        if n_samples < self.svd_dim:
            pad_rows = np.zeros((self.svd_dim - n_samples, dense.shape[1]))
            dense = np.vstack([dense, pad_rows])
        if dense.shape[1] < self.svd_dim:
            pad_cols = np.zeros((dense.shape[0], self.svd_dim - dense.shape[1]))
            dense = np.hstack([dense, pad_cols])
        self.svd.fit(dense)
        self.fitted = True
        return self

    def _pad_tfidf(self, matrix) -> np.ndarray:
        """Pad TF-IDF matrix to match SVD expected feature count."""
        dense = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
        # SVD was fit on padded data, so expected features is svd_dim
        expected = self.svd.components_.shape[1]
        if dense.shape[1] < expected:
            pad_cols = np.zeros((dense.shape[0], expected - dense.shape[1]))
            dense = np.hstack([dense, pad_cols])
        return dense

    def transform(self, aggregated_stock_news: pd.DataFrame) -> pd.DataFrame:
        """Transform aggregated stock news to feature columns.

        Expects input from aggregate_stock_news() with columns:
          [trade_date, ts_code, news_count, title_count,
           avg_content_length, aggregated_text]

        Returns DataFrame:
          [trade_date, ts_code, stock_news_count, stock_title_count,
           stock_avg_content_length, stock_news_tfidf_00..15]
        """
        output_columns = (
            [TRADE_DATE, TS_CODE]
            + STOCK_NEWS_STAT_COLUMNS
            + self.feature_names
        )
        if not self.fitted:
            raise RuntimeError("StockNewsFeatureGenerator must be fit before transform")

        if aggregated_stock_news.empty:
            return pd.DataFrame(columns=output_columns)

        base = aggregated_stock_news.copy()
        texts = base["aggregated_text"].fillna("").astype(str)
        tfidf_matrix = self.vectorizer.transform(texts)
        dense = self._pad_tfidf(tfidf_matrix)
        reduced = self.svd.transform(dense)

        features = pd.DataFrame(reduced, columns=self.feature_names)

        stat_values = base[["news_count", "title_count", "avg_content_length"]].copy()
        stat_values.columns = STOCK_NEWS_STAT_COLUMNS

        result = pd.concat(
            [
                base[[TRADE_DATE, TS_CODE]].reset_index(drop=True),
                stat_values.reset_index(drop=True),
                features,
            ],
            axis=1,
        )
        return result

    def fit_transform(self, aggregated_stock_news: pd.DataFrame) -> pd.DataFrame:
        return self.fit(aggregated_stock_news).transform(aggregated_stock_news)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "StockNewsFeatureGenerator":
        return joblib.load(path)
