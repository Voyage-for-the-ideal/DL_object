from __future__ import annotations

import pandas as pd

from src.data.news_features import aggregate_news
from src.features.news_features import NewsTfidfFeatureGenerator


def test_tfidf_vocab_fit_only_on_train() -> None:
    train = pd.DataFrame(
        {
            "trade_date": ["20200101"],
            "news_count": [1],
            "title_count": [1],
            "avg_content_length": [5],
            "aggregated_text": ["trainword"],
        }
    )
    valid = pd.DataFrame(
        {
            "trade_date": ["20200102"],
            "news_count": [1],
            "title_count": [1],
            "avg_content_length": [5],
            "aggregated_text": ["validationonly"],
        }
    )
    generator = NewsTfidfFeatureGenerator(max_features=4).fit(train)
    assert "validationonly" not in generator.vectorizer.vocabulary_
    transformed = generator.transform(valid)
    assert transformed.filter(like="news_tfidf_").shape[1] == 4


def test_missing_news_outputs_zero_stats() -> None:
    aggregated = aggregate_news(pd.DataFrame(), date="20200101")
    assert aggregated.loc[0, "news_count"] == 0
