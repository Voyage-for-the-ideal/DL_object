"""Small raw-data feature pipeline used by smoke tests and daily prediction."""

from __future__ import annotations

import pandas as pd

from src.data.calendar import TradingCalendar
from src.data.loader import CsvDataLoader
from src.data.schema import TRADE_DATE
from src.data.universe import UniverseBuilder
from src.features.merge import merge_feature_frames
from src.features.metric_features import build_metric_features
from src.features.moneyflow_features import build_moneyflow_features
from src.features.price_features import build_price_features


def build_feature_panel(
    loader: CsvDataLoader,
    trade_dates: list[str],
    universe_mode: str = "official",
    windows: tuple[int, ...] = (5, 10, 20),
    min_amount: float = 0.0,
) -> pd.DataFrame:
    """Build a merged feature panel for supplied signal dates."""
    daily = _concat([loader.load_daily(date) for date in trade_dates])
    metric = _concat([loader.load_metric(date) for date in trade_dates])
    moneyflow = _concat([loader.load_moneyflow(date) for date in trade_dates])
    panel = merge_feature_frames(
        [
            build_price_features(daily, windows=windows),
            build_metric_features(metric),
            build_moneyflow_features(moneyflow, windows=windows),
        ]
    )
    if panel.empty:
        return panel
    universe = UniverseBuilder(loader, min_amount=min_amount)
    allowed_rows: list[pd.DataFrame] = []
    for trade_date, group in panel.groupby(TRADE_DATE):
        codes = universe.get_tradeable_universe(str(trade_date), mode=universe_mode)
        allowed_rows.append(group[group["ts_code"].astype(str).isin(codes)])
    return _concat(allowed_rows).reset_index(drop=True)


def build_latest_feature_frame(
    loader: CsvDataLoader,
    calendar: TradingCalendar,
    signal_date: str,
    lookback: int,
    universe_mode: str = "official",
) -> pd.DataFrame:
    """Build only the latest signal-date rows, reading no future daily files."""
    all_dates = calendar.trade_dates_between(calendar.trade_dates[0], signal_date)
    dates = all_dates[-lookback:] if lookback > 0 else [signal_date]
    panel = build_feature_panel(loader, dates, universe_mode=universe_mode)
    return panel[panel[TRADE_DATE].astype(str) == signal_date].reset_index(drop=True)


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
