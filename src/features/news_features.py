"""Market-level TF-IDF and FinBERT news features."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.news_features import aggregate_news
from src.data.schema import TRADE_DATE, normalize_trade_date


class NewsTfidfFeatureGenerator:
    """Fit TF-IDF only on training-period aggregated news text."""

    def __init__(self, max_features: int = 128) -> None:
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.fitted = False

    @property
    def feature_names(self) -> list[str]:
        return [f"news_tfidf_{index:03d}" for index in range(self.max_features)]

    def fit(self, aggregated_news: pd.DataFrame) -> "NewsTfidfFeatureGenerator":
        texts = aggregated_news.get("aggregated_text", pd.Series(dtype=str)).fillna("").astype(str)
        if texts.str.len().sum() == 0:
            self.vectorizer.fit(["empty"])
        else:
            self.vectorizer.fit(texts)
        self.fitted = True
        return self

    def transform(
        self,
        aggregated_news: pd.DataFrame,
        dates: list[str] | None = None,
    ) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("NewsTfidfFeatureGenerator must be fit before transform")
        base = _ensure_news_rows(aggregated_news, dates)
        texts = base["aggregated_text"].fillna("").astype(str)
        matrix = self.vectorizer.transform(texts).toarray()
        if matrix.shape[1] < self.max_features:
            padding = np.zeros((matrix.shape[0], self.max_features - matrix.shape[1]))
            matrix = np.hstack([matrix, padding])
        tfidf = pd.DataFrame(matrix[:, : self.max_features], columns=self.feature_names)
        return pd.concat(
            [base.drop(columns=["aggregated_text"]).reset_index(drop=True), tfidf], axis=1
        )

    def fit_transform(self, aggregated_news: pd.DataFrame) -> pd.DataFrame:
        return self.fit(aggregated_news).transform(aggregated_news)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "NewsTfidfFeatureGenerator":
        return joblib.load(path)


class NewsFinbertFeatureGenerator:
    """FinBERT embeddings + TruncatedSVD for market-level news features.

    Fit FinBERT on training-period aggregated news to extract 768-dim embeddings,
    then fit TruncatedSVD to compress to svd_dim. BERT weights are NOT saved —
    only SVD + model_name are serialized; the BERT model is re-downloaded on load.
    """

    def __init__(
        self,
        model_name: str = "yiyanghkust/finbert-tone-chinese",
        svd_dim: int = 16,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.svd_dim = svd_dim
        self.device = device
        self.svd = TruncatedSVD(n_components=svd_dim, random_state=42)
        self.fitted = False
        self._tokenizer: object = None
        self._model: object = None

    @property
    def feature_names(self) -> list[str]:
        return [f"finbert_{i:02d}" for i in range(self.svd_dim)]

    def _load_bert(self) -> None:
        from transformers import AutoModel, AutoTokenizer  # type: ignore[import-untyped]

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.eval()
        self._model.to(self.device)

    def _encode_texts(self, texts: pd.Series) -> np.ndarray:
        if self._tokenizer is None or self._model is None:
            self._load_bert()

        text_list = texts.fillna("").astype(str).tolist()
        all_embeddings: list[np.ndarray] = []
        batch_size = 32

        import torch

        with torch.no_grad():
            for i in range(0, len(text_list), batch_size):
                batch = text_list[i : i + batch_size]
                inputs = self._tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self._model(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                all_embeddings.append(emb)

        return np.vstack(all_embeddings)

    def fit(self, aggregated_news: pd.DataFrame) -> "NewsFinbertFeatureGenerator":
        texts = aggregated_news.get("aggregated_text", pd.Series(dtype=str))
        embeddings = self._encode_texts(texts)
        self.svd.fit(embeddings)
        self.fitted = True
        return self

    def transform(
        self,
        aggregated_news: pd.DataFrame,
        dates: list[str] | None = None,
    ) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("NewsFinbertFeatureGenerator must be fit before transform")
        base = _ensure_news_rows(aggregated_news, dates)
        texts = base["aggregated_text"]
        embeddings = self._encode_texts(texts)
        reduced = self.svd.transform(embeddings)
        features = pd.DataFrame(reduced, columns=self.feature_names)
        return pd.concat(
            [base.drop(columns=["aggregated_text"]).reset_index(drop=True), features],
            axis=1,
        )

    def fit_transform(self, aggregated_news: pd.DataFrame) -> pd.DataFrame:
        return self.fit(aggregated_news).transform(aggregated_news)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model_name": self.model_name,
                "svd_dim": self.svd_dim,
                "svd": self.svd,
                "fitted": self.fitted,
            },
            target,
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "NewsFinbertFeatureGenerator":
        data = joblib.load(path)
        instance = cls(
            model_name=data["model_name"],
            svd_dim=data["svd_dim"],
        )
        instance.svd = data["svd"]
        instance.fitted = data["fitted"]
        return instance


def _ensure_news_rows(aggregated_news: pd.DataFrame, dates: list[str] | None) -> pd.DataFrame:
    columns = [TRADE_DATE, "news_count", "title_count", "avg_content_length", "aggregated_text"]
    if aggregated_news.empty:
        frame = pd.DataFrame(columns=columns)
    else:
        frame = aggregated_news.copy()
    if dates is not None:
        normalized = [normalize_trade_date(date) for date in dates]
        frame = frame.set_index(TRADE_DATE) if TRADE_DATE in frame.columns else pd.DataFrame()
        rows: list[dict[str, object]] = []
        for date in normalized:
            if not frame.empty and date in frame.index:
                row = frame.loc[date]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                rows.append({TRADE_DATE: date, **row.to_dict()})
            else:
                rows.append(
                    {
                        TRADE_DATE: date,
                        "news_count": 0,
                        "title_count": 0,
                        "avg_content_length": 0.0,
                        "aggregated_text": "",
                    }
                )
        return pd.DataFrame(rows, columns=columns)
    for column in columns:
        if column not in frame.columns:
            frame[column] = "" if column == "aggregated_text" else 0
    return frame[columns]


def aggregate_raw_news(news: pd.DataFrame, date: str | None = None) -> pd.DataFrame:
    return aggregate_news(news, date=date)
