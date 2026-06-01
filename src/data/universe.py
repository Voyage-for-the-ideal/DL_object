"""Daily universe construction and filtering."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.loader import CsvDataLoader
from src.data.schema import TRADE_DATE, TS_CODE, normalize_trade_date


@dataclass
class UniverseBuilder:
    loader: CsvDataLoader
    min_amount: float = 0.0
    min_amount_pct: float = 0.0
    exclude_st: bool = True
    exclude_bse: bool = True

    def get_hs300_universe(self, trade_date: str) -> set[str]:
        weights = self.loader.load_index_weight(trade_date=trade_date, index_code="000300.SH")
        if weights.empty:
            return set()
        return set(weights["con_code"].astype(str))

    def _basic_universe(self) -> set[str]:
        basic = self.loader.load_basic()
        if basic.empty:
            return set()
        frame = basic.copy()
        if self.exclude_bse:
            market = frame.get("market", pd.Series("", index=frame.index)).astype(str)
            code = frame[TS_CODE].astype(str)
            allowed = ~market.str.contains("北交", na=False) & ~code.str.endswith(".BJ")
            return set(frame.loc[allowed, TS_CODE].astype(str))
        return set(frame[TS_CODE].astype(str))

    def get_st_universe(self, trade_date: str) -> set[str]:
        target = normalize_trade_date(trade_date)
        st_date = self.loader.latest_partition_date("stock_st", max_date=target)
        if st_date is None:
            return set()
        st = self.loader.load_st(st_date)
        if st.empty:
            return set()
        st = st[st[TRADE_DATE] <= target] if TRADE_DATE in st.columns else st
        return set(st[TS_CODE].astype(str))

    def get_official_universe(self, trade_date: str) -> set[str]:
        universe = self._basic_universe()
        if self.exclude_st:
            universe = universe - self.get_st_universe(trade_date)
        return universe

    def get_tradeable_universe(self, trade_date: str, mode: str = "official") -> set[str]:
        if mode == "hs300":
            base = self.get_hs300_universe(trade_date)
        elif mode == "official":
            base = self.get_official_universe(trade_date)
        elif mode == "all":
            base = set(self.loader.load_basic().get(TS_CODE, pd.Series(dtype=str)).astype(str))
        else:
            raise ValueError(f"Unsupported universe mode: {mode}")
        return self.filter_tradeable(trade_date, base)

    def filter_tradeable(self, trade_date: str, universe: set[str]) -> set[str]:
        daily = self.loader.load_daily(trade_date)
        if daily.empty:
            return set()
        frame = daily[daily[TS_CODE].astype(str).isin(universe)].copy()
        vol = pd.to_numeric(frame.get("vol", 0), errors="coerce").fillna(0)
        amount = pd.to_numeric(frame.get("amount", 0), errors="coerce").fillna(0)
        frame = frame[(vol > 0) & (amount >= self.min_amount)]
        if self.min_amount_pct > 0 and len(frame) > 0:
            threshold = frame["amount"].quantile(self.min_amount_pct)
            frame = frame[frame["amount"] >= threshold]
        return set(frame[TS_CODE].astype(str))
