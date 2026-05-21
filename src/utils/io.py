"""Small file IO helpers used across the project."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(data: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    return target


def write_csv(frame: pd.DataFrame, path: str | Path, **kwargs: Any) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    frame.to_csv(target, index=False, **kwargs)
    return target


def save_yaml(data: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)
    return target


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def make_run_dir(outputs_root: str | Path, run_id: str | None = None) -> Path:
    run_name = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    return ensure_dir(Path(outputs_root) / "runs" / run_name)


def ensure_output_tree(outputs_root: str | Path) -> dict[str, Path]:
    root = ensure_dir(outputs_root)
    names = ["runs", "evaluation", "backtest", "signals", "orders", "cache"]
    return {name: ensure_dir(root / name) for name in names}
