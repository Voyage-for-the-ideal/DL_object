"""Multi-horizon alpha factors for cross-sectional stock ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.data.schema import TRADE_DATE, TS_CODE


def _safe_divide(a: pd.Series, b: pd.Series, fill: float = 0.0) -> pd.Series:
    result = a / b.replace(0, np.nan)
    return result.fillna(fill).replace([np.inf, -np.inf], fill)


def _grouped_pct_change(series: pd.Series, group: pd.Series, periods: int) -> pd.Series:
    return series.groupby(group).pct_change(periods).reset_index(level=0, drop=True)


def _grouped_shift(series: pd.Series, group: pd.Series, periods: int) -> pd.Series:
    return series.groupby(group).shift(periods).reset_index(level=0, drop=True)


def build_alpha_factors(
    daily: pd.DataFrame,
    metric: pd.DataFrame,
    moneyflow: pd.DataFrame,
    basic: pd.DataFrame | None = None,
    windows: tuple[int, ...] = (5, 10, 20),
) -> pd.DataFrame:
    """
    Build ~200 alpha factors from daily OHLCV, metric, and moneyflow data.

    Returns a DataFrame with columns [trade_date, ts_code, ...alpha_factors...].
    """
    factors: list[pd.DataFrame] = []

    # ---- 1. Price-derived factors ----
    price = daily.copy()
    price[TRADE_DATE] = price[TRADE_DATE].astype(str)
    price = price.sort_values([TS_CODE, TRADE_DATE]).reset_index(drop=True)
    grp = price[TS_CODE]

    # Basic returns
    price["ret_1d"] = price.groupby(TS_CODE)["close"].pct_change()
    price["ret_overnight"] = _safe_divide(price["open"] - price["pre_close"], price["pre_close"])
    price["ret_intraday"] = _safe_divide(price["close"] - price["open"], price["open"])
    price["hl_range"] = _safe_divide(price["high"] - price["low"], price["close"])
    price["gap"] = _safe_divide(price["open"] - price["pre_close"], price["pre_close"])

    # Volume/amount indicators
    price["vol_chg"] = price.groupby(TS_CODE)["vol"].pct_change()
    price["amount_chg"] = price.groupby(TS_CODE)["amount"].pct_change()
    price["turnover_proxy"] = _safe_divide(
        price["vol"],
        price["vol"].groupby(grp).transform(lambda x: x.rolling(20, min_periods=5).mean()),
    )
    price["amount_proxy"] = _safe_divide(
        price["amount"],
        price["amount"].groupby(grp).transform(lambda x: x.rolling(20, min_periods=5).mean()),
    )

    # Multi-horizon returns
    for h in [1, 5, 10, 20]:
        price[f"ret_{h}d"] = price.groupby(TS_CODE)["close"].pct_change(h)
        price[f"vol_chg_{h}d"] = price.groupby(TS_CODE)["vol"].pct_change(h)

    # Rolling statistics of ret_1d
    for w in windows:
        g = price.groupby(TS_CODE)["ret_1d"]
        price[f"ret_mean_{w}d"] = g.transform(
            lambda x: x.rolling(w, min_periods=max(3, w // 2)).mean()
        )
        price[f"ret_std_{w}d"] = g.transform(
            lambda x: x.rolling(w, min_periods=max(3, w // 2)).std()
        )
        price[f"ret_sharpe_{w}d"] = _safe_divide(price[f"ret_mean_{w}d"], price[f"ret_std_{w}d"])
        price[f"ret_skew_{w}d"] = g.transform(
            lambda x: x.rolling(w, min_periods=max(5, w // 2)).skew()
        )
        # Downside volatility (only negative returns)
        neg = price["ret_1d"].clip(upper=0)
        price[f"downside_vol_{w}d"] = neg.groupby(grp).transform(
            lambda x: x.rolling(w, min_periods=max(3, w // 2)).std()
        )
    # Cumulative return for max drawdown
    price["cumret"] = (1 + price["ret_1d"].fillna(0)).groupby(grp).cumprod()
    # Max drawdown
    for w in windows:
        rolling_max = price.groupby(TS_CODE)["cumret"].transform(
            lambda x: x.rolling(w, min_periods=max(3, w // 2)).max()
        )
        price[f"max_dd_{w}d"] = _safe_divide(rolling_max, price["cumret"]) - 1

    # Volume-price correlation
    for w in windows:
        g = price.groupby(TS_CODE)
        price[f"vol_price_corr_{w}d"] = (
            g.apply(
                lambda x: x["vol"].rolling(w, min_periods=max(5, w // 2)).corr(x["close"])
            )
            .reset_index(level=0, drop=True)
        )

    # Price features to keep
    price_cols = [TRADE_DATE, TS_CODE] + [
        c
        for c in price.columns
        if c
        not in [
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "change",
            "pct_chg",
            "vol",
            "amount",
            "vwap",
            TRADE_DATE,
            TS_CODE,
        ]
    ]
    price_cols = [c for c in price_cols if c in price.columns]
    factors.append(price[price_cols])

    # ---- 2. Metric-derived factors ----
    if metric is not None and not metric.empty:
        met = metric.copy()
        met[TRADE_DATE] = met[TRADE_DATE].astype(str)
        met = met.sort_values([TS_CODE, TRADE_DATE]).reset_index(drop=True)
        met_grp = met[TS_CODE]

        # Raw metric values with log transforms
        for col in [
            "pe",
            "pe_ttm",
            "pb",
            "ps",
            "ps_ttm",
            "total_mv",
            "circ_mv",
            "turnover_rate",
            "turnover_rate_f",
            "volume_ratio",
            "dv_ratio",
            "dv_ttm",
        ]:
            if col in met.columns:
                met[f"metric_{col}"] = pd.to_numeric(met[col], errors="coerce")

        if "total_mv" in met.columns:
            met["metric_total_mv_log"] = np.log1p(pd.to_numeric(met["total_mv"], errors="coerce"))
        if "circ_mv" in met.columns:
            met["metric_circ_mv_log"] = np.log1p(pd.to_numeric(met["circ_mv"], errors="coerce"))

        # Metric changes over horizons
        for col in ["pe", "pe_ttm", "pb", "turnover_rate", "turnover_rate_f", "volume_ratio"]:
            mc = f"metric_{col}"
            if mc in met.columns:
                for h in [5, 20]:
                    met[f"{mc}_chg_{h}d"] = met.groupby(TS_CODE)[mc].pct_change(h)
                for w in windows:
                    met[f"{mc}_mean_{w}d"] = met.groupby(TS_CODE)[mc].transform(
                        lambda x: x.rolling(w, min_periods=max(3, w // 2)).mean()
                    )

        met_cols = [TRADE_DATE, TS_CODE] + [
            c for c in met.columns if c.startswith("metric_") and c in met.columns
        ]
        factors.append(met[met_cols])

    # ---- 3. Moneyflow-derived factors ----
    if moneyflow is not None and not moneyflow.empty:
        mf = moneyflow.copy()
        mf[TRADE_DATE] = mf[TRADE_DATE].astype(str)
        mf = mf.sort_values([TS_CODE, TRADE_DATE]).reset_index(drop=True)

        # Net flows per order size level
        for level in ["sm", "md", "lg", "elg"]:
            buy_vol = pd.to_numeric(mf.get(f"buy_{level}_vol", pd.Series(dtype=float)), errors="coerce").fillna(0)
            sell_vol = pd.to_numeric(mf.get(f"sell_{level}_vol", pd.Series(dtype=float)), errors="coerce").fillna(0)
            total = (buy_vol + sell_vol).replace(0, np.nan)
            mf[f"mf_{level}_net_vol"] = buy_vol - sell_vol
            mf[f"mf_{level}_imbalance"] = (buy_vol - sell_vol) / total

        # Total net flow
        if "net_mf_vol" in mf.columns:
            mf["mf_net_vol"] = pd.to_numeric(mf["net_mf_vol"], errors="coerce").fillna(0)
        if "net_mf_amount" in mf.columns:
            mf["mf_net_amount"] = pd.to_numeric(mf["net_mf_amount"], errors="coerce").fillna(0)

        # Large-order dominance
        for level in ["lg", "elg"]:
            buy_col = f"mf_{level}_net_vol"
            if buy_col in mf.columns and "mf_net_vol" in mf.columns:
                mf[f"mf_{level}_dominance"] = _safe_divide(
                    mf[buy_col].abs(), mf["mf_net_vol"].abs() + 1
                )

        # Consecutive net inflow days
        if "mf_net_vol" in mf.columns:
            mf["mf_net_inflow"] = (mf["mf_net_vol"] > 0).astype(int)
            mf["mf_consec_inflow"] = mf.groupby(TS_CODE)["mf_net_inflow"].transform(
                lambda x: x.rolling(10, min_periods=1).sum()
            )

        # Rolling moneyflow stats
        for w in windows:
            if "mf_net_vol" in mf.columns:
                mf[f"mf_net_vol_mean_{w}d"] = mf.groupby(TS_CODE)["mf_net_vol"].transform(
                    lambda x: x.rolling(w, min_periods=max(3, w // 2)).mean()
                )
                mf[f"mf_net_vol_accel_{w}d"] = mf[f"mf_net_vol_mean_{w}d"] - mf.groupby(
                    TS_CODE
                )["mf_net_vol"].transform(
                    lambda x: x.shift(w).rolling(w, min_periods=max(3, w // 2)).mean()
                )

        mf_cols = [TRADE_DATE, TS_CODE] + [
            c for c in mf.columns if c.startswith("mf_") and c in mf.columns
        ]
        factors.append(mf[mf_cols])

    # ---- 4. Cross-sectional features ----
    # Per-date rank percentiles for key metrics
    cs_features: list[pd.DataFrame] = []
    for factor_df in factors:
        if TRADE_DATE not in factor_df.columns or TS_CODE not in factor_df.columns:
            continue
        cs_frame = factor_df[[TRADE_DATE, TS_CODE]].copy()
        cs_cols = [
            c
            for c in factor_df.columns
            if c not in [TRADE_DATE, TS_CODE] and not c.startswith("cs_")
        ]
        for col in cs_cols[:50]:  # limit to avoid explosion
            cs_frame[f"cs_rank_{col}"] = factor_df.groupby(TRADE_DATE)[col].rank(pct=True)
        cs_features.append(cs_frame)

    # ---- 5. Momentum/reversal hybrids ----
    import re
    ret_cols = [c for c in price.columns if re.match(r"^ret_\d+d$", c)]
    hybrid_features: list[pd.DataFrame] = []
    for i, c1 in enumerate(ret_cols):
        for c2 in ret_cols[i + 1 :]:
            # Only cross short-term with long-term
            h1 = int(c1.replace("ret_", "").replace("d", ""))
            h2 = int(c2.replace("ret_", "").replace("d", ""))
            if h1 <= 5 and h2 >= 10:
                col_name = f"hybrid_ret{h1}d_x_ret{h2}d"
                tmp = price[[TRADE_DATE, TS_CODE]].copy()
                tmp[col_name] = price[c1].fillna(0) * price[c2].fillna(0)
                hybrid_features.append(tmp)
                break  # one interaction per short-term ret is enough
        if len(hybrid_features) >= 3:
            break

    # ---- Merge all factor groups ----
    # Start with price factors
    result = factors[0]
    for df in factors[1:]:
        result = result.merge(df, on=[TRADE_DATE, TS_CODE], how="left")

    # Add cross-sectional features
    for df in cs_features:
        cs_cols = [c for c in df.columns if c.startswith("cs_")]
        if cs_cols:
            result = result.merge(
                df[[TRADE_DATE, TS_CODE] + cs_cols], on=[TRADE_DATE, TS_CODE], how="left"
            )

    # Add hybrid features
    for df in hybrid_features:
        result = result.merge(df, on=[TRADE_DATE, TS_CODE], how="left")

    # Add industry-relative features if basic data is provided
    if basic is not None and not basic.empty:
        industry_map = (
            basic.set_index(TS_CODE).get("industry", pd.Series(dtype=str)).to_dict()
            if "industry" in basic.columns
            else {}
        )
        market_map = (
            basic.set_index(TS_CODE).get("market", pd.Series(dtype=str)).to_dict()
            if "market" in basic.columns
            else {}
        )
        group_map = industry_map if industry_map else market_map
        if group_map:
            result["_group"] = result[TS_CODE].map(group_map).fillna("unknown")
            key_cols = ["pe", "pb", "turnover_rate"]
            for col in key_cols:
                mc = f"metric_{col}"
                if mc in result.columns:
                    gmean = result.groupby([TRADE_DATE, "_group"])[mc].transform("mean")
                    gstd = (
                        result.groupby([TRADE_DATE, "_group"])[mc]
                        .transform("std")
                        .replace(0, 1)
                    )
                    result[f"ind_rel_{col}"] = (result[mc] - gmean) / gstd
            result = result.drop(columns=["_group"])

    return result.reset_index(drop=True)
