"""Lightweight feature cache helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.io import ensure_dir, write_json


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


class FeatureStore:
    def __init__(self, root: str | Path, config: dict[str, Any]) -> None:
        self.root = ensure_dir(Path(root) / config_hash(config))

    def path_for_date(self, trade_date: str) -> Path:
        return self.root / f"{trade_date}.csv"

    def save_date(self, trade_date: str, frame: pd.DataFrame) -> Path:
        path = self.path_for_date(trade_date)
        frame.to_csv(path, index=False)
        return path

    def load_date(self, trade_date: str) -> pd.DataFrame:
        path = self.path_for_date(trade_date)
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    def save_metadata(self, metadata: dict[str, Any]) -> Path:
        return write_json(metadata, self.root / "metadata.json")
