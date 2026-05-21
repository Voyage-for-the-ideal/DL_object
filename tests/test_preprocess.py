from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.preprocess import TrainOnlyPreprocessor


def test_validation_extreme_does_not_change_fit_params() -> None:
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    valid = pd.DataFrame({"x": [999.0]})
    pre = TrainOnlyPreprocessor(lower_quantile=0.0, upper_quantile=1.0).fit(train, ["x"])
    assert pre.means is not None
    mean_before = pre.means.copy()
    pre.transform(valid)
    pd.testing.assert_series_equal(pre.means, mean_before)


def test_preprocessor_save_load_roundtrip(tmp_path: Path) -> None:
    frame = pd.DataFrame({"x": [1.0, None, 3.0]})
    pre = TrainOnlyPreprocessor().fit(frame, ["x"])
    path = pre.save(tmp_path / "pre.joblib")
    loaded = TrainOnlyPreprocessor.load(path)
    pd.testing.assert_frame_equal(pre.transform(frame), loaded.transform(frame))
