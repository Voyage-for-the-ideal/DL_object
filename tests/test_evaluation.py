from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.metrics import daily_ic, evaluate_predictions


def test_perfect_positive_and_negative_ic() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": ["20200101"] * 3,
            "ts_code": ["A", "B", "C"],
            "score": [1.0, 2.0, 3.0],
            "label": [1.0, 2.0, 3.0],
        }
    )
    assert daily_ic(frame).loc[0, "ic"] == pytest.approx(1.0)
    frame["score"] = [3.0, 2.0, 1.0]
    assert daily_ic(frame).loc[0, "ic"] == pytest.approx(-1.0)


def test_too_few_samples_are_skipped() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": ["20200101"] * 2,
            "ts_code": ["A", "B"],
            "score": [1.0, 2.0],
            "label": [1.0, 2.0],
        }
    )
    assert daily_ic(frame).empty


def test_evaluate_predictions_fields() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": ["20200101"] * 5,
            "ts_code": list("ABCDE"),
            "score": [1, 2, 3, 4, 5],
            "label": [1, 2, 3, 4, 5],
        }
    )
    metrics, _, groups = evaluate_predictions(frame)
    assert "icir" in metrics
    assert not groups.empty
