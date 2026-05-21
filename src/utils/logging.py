"""Logging setup for CLI and experiment runs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.utils.io import ensure_dir


def get_logger(name: str = "dl_object") -> logging.Logger:
    return logging.getLogger(name)


def setup_logging(
    log_dir: str | Path | None = None,
    run_id: str | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    logger = get_logger()
    logger.setLevel(level)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_dir is not None:
        target_dir = ensure_dir(log_dir)
        log_name = f"{run_id or 'run'}.log"
        file_handler = logging.FileHandler(target_dir / log_name, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def log_config_summary(logger: logging.Logger, config: dict[str, Any]) -> None:
    data = config.get("data", {})
    model = config.get("model", {})
    logger.info(
        "config: data=%s..%s valid=%s..%s model=%s",
        data.get("start_date"),
        data.get("train_end_date"),
        data.get("valid_start_date"),
        data.get("valid_end_date"),
        model.get("name"),
    )
