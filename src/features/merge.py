"""Feature table merging helpers."""

from __future__ import annotations

import pandas as pd

from src.data.schema import TRADE_DATE, TS_CODE


def merge_feature_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame(columns=[TRADE_DATE, TS_CODE])
    result = non_empty[0].copy()
    for frame in non_empty[1:]:
        if TS_CODE in frame.columns:
            result = result.merge(frame, on=[TRADE_DATE, TS_CODE], how="left")
        else:
            result = result.merge(frame, on=[TRADE_DATE], how="left")
    return result
