"""Extract A-share stock codes from news text via regex + name matching."""

from __future__ import annotations

import re
from typing import Tuple

import pandas as pd

from src.data.schema import TS_CODE

_FULLWIDTH_MAP = str.maketrans(
    "ＡＢＣＤＥＦＧＨＩＪ"
    "ＫＬＭＮＯＰＱＲＳ"
    "ＴＵＶＷＸＹＺ"
    "ａｂｃｄｅｆｇｈｉｊ"
    "ｋｌｍｎｏｐｑｒｓ"
    "ｔｕｖｗｘｙｚ"
    "０１２３４５６７８９",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
)

_NAME_MIN_LENGTH = 3

# Group 1: name (Chinese chars + alphanumeric), Group 2: 6-digit bare code or full ts_code
_CODE_PATTERN = re.compile(
    r"([一-鿿\w]+)\((\d{6}(?:\.(?:SZ|SH|BJ))?)\)"
)


def _normalize_name(name: str) -> str:
    """Convert full-width letters/numbers to half-width."""
    normalized = name.translate(_FULLWIDTH_MAP)
    return normalized.strip()


def build_stock_lookup(basic: pd.DataFrame) -> dict:
    """Build lookup structures from basic.csv for stock code extraction.

    Returns dict with:
      - valid_codes: set of all bare 6-digit symbols known to basic.csv
      - symbol_to_code: dict mapping 6-digit symbol to ts_code
      - name_to_code: dict mapping normalized stock name to ts_code
      - sorted_names: list of (name, code) sorted by name length desc (greedy matching)
    """
    frame = basic.copy()
    frame[TS_CODE] = frame[TS_CODE].astype(str)
    symbol_col = frame.get("symbol", pd.Series(dtype=str)).astype(str)
    name_col = frame.get("name", pd.Series("", index=frame.index)).astype(str)

    valid_codes: set[str] = set()
    symbol_to_code: dict[str, str] = {}
    # Bare codes are unique enough for A-shares; last one wins if duplicate
    for _, row in frame.iterrows():
        code = str(row[TS_CODE])
        sym = str(row["symbol"]) if "symbol" in row else code[:6]
        valid_codes.add(sym)
        symbol_to_code[sym] = code

    name_to_code: dict[str, str] = {}
    names: list[tuple[str, str]] = []
    for _, row in frame.iterrows():
        name = str(row["name"]) if "name" in row else ""
        if not name:
            continue
        normalized = _normalize_name(name)
        if len(normalized) < _NAME_MIN_LENGTH:
            continue
        code = str(row[TS_CODE])
        name_to_code[normalized] = code
        names.append((normalized, code))

    # Sort by length descending for greedy longest-match-first
    names.sort(key=lambda x: len(x[0]), reverse=True)

    return {
        "valid_codes": valid_codes,
        "symbol_to_code": symbol_to_code,
        "name_to_code": name_to_code,
        "sorted_names": names,
    }


def extract_stock_codes(news: pd.DataFrame, lookup: dict) -> pd.DataFrame:
    """Add ts_code column to raw news rows by extracting stock codes from text.

    Strategy (applied per row):
    1. Search title+content for '公司名(6位代码)' regex. Validate code against
       lookup['valid_codes']. Resolve bare 6-digit codes via symbol_to_code.
    2. For remaining unmatched rows, search title via longest-name-first greedy
       matching against lookup['sorted_names'].

    Returns a copy of news with added ts_code column ("" for unmatched rows).
    """
    result = news.copy()
    ts_codes: list[str] = []

    valid_codes = lookup["valid_codes"]
    symbol_to_code = lookup["symbol_to_code"]
    sorted_names = lookup["sorted_names"]

    for _, row in result.iterrows():
        title = str(row.get("title", ""))
        content = str(row.get("content", ""))
        combined = f"{title} {content}"

        # Step 1: regex for 公司名(代码)
        matched_code = ""
        for match in _CODE_PATTERN.finditer(combined):
            captured_code = match.group(2)
            # Normalize: strip exchange suffix for validation
            bare = captured_code[:6]
            if bare in valid_codes:
                matched_code = symbol_to_code.get(bare, captured_code)
                break

        # Step 2: name matching (fallback)
        if not matched_code:
            title_normalized = _normalize_name(title)
            for name, code in sorted_names:
                if name in title_normalized:
                    matched_code = code
                    break

        ts_codes.append(matched_code)

    result[TS_CODE] = ts_codes
    return result


def split_market_stock_news(
    news: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split news into market-level (unmatched) and stock-level (matched).

    Args:
        news: DataFrame with ts_code column from extract_stock_codes.

    Returns:
        (market_news, stock_news) — market rows have ts_code=="",
        stock rows have ts_code!="".
    """
    if TS_CODE not in news.columns:
        return news.copy(), pd.DataFrame(columns=news.columns.tolist())
    mask = news[TS_CODE].astype(str).str.strip() != ""
    market = news[~mask].copy()
    stock = news[mask].copy()
    return market, stock
