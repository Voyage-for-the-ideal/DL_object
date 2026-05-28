"""YAML configuration loading and validation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml

Config = Dict[str, Any]


def find_project_root(start: Path | None = None) -> Path:
    """Find the workspace root by walking upward from *start*."""
    current = (start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for parent in (current, *current.parents):
        if (parent / "CLAUDE.md").exists() or (parent / "大作业.md").exists():
            return parent
    return current


def _deep_merge(base: Config, override: Config) -> Config:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _set_by_dotted_key(config: Config, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor: Config = config
    for part in parts[:-1]:
        child = cursor.setdefault(part, {})
        if not isinstance(child, dict):
            msg = f"Cannot set override {dotted_key!r}: {part!r} is not a mapping"
            raise ValueError(msg)
        cursor = child
    cursor[parts[-1]] = value


def _coerce_scalar(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def apply_overrides(config: Config, overrides: dict[str, Any] | None = None) -> Config:
    """Apply dotted-key overrides such as ``training.epochs=1``."""
    result = deepcopy(config)
    for key, value in (overrides or {}).items():
        _set_by_dotted_key(result, key, _coerce_scalar(value) if isinstance(value, str) else value)
    return result


def resolve_config_paths(config: Config, project_root: Path | None = None) -> Config:
    """Resolve path-like config entries against the project root."""
    root = project_root or find_project_root()
    result = deepcopy(config)
    for section, key in (("data", "root"), ("outputs", "root")):
        raw = Path(str(result[section][key]))
        result[section][key] = str(raw if raw.is_absolute() else (root / raw).resolve())
    result["project_root"] = str(root)
    return result


def validate_config(config: Config) -> None:
    """Validate the core fields needed by every pipeline."""
    data = config.get("data", {})
    dates = [
        data.get("start_date"),
        data.get("train_end_date"),
        data.get("valid_start_date"),
        data.get("valid_end_date"),
    ]
    if any(date is None for date in dates):
        raise ValueError("Missing one or more required data date fields")
    if not (dates[0] <= dates[1] < dates[2] <= dates[3]):
        raise ValueError("Date order must satisfy start <= train_end < valid_start <= valid_end")
    universe_mode = data.get("universe_mode")
    if universe_mode not in {"hs300", "official", "all"}:
        raise ValueError(f"Unsupported universe_mode: {universe_mode!r}")
    model_name = config.get("model", {}).get("name")
    allowed_models = {"ridge", "elasticnet", "lightgbm", "gbdt", "mlp", "transformer_encoder", "ft_transformer"}
    if model_name not in allowed_models:
        raise ValueError(f"Unsupported model.name: {model_name!r}")
    if not config.get("outputs", {}).get("root"):
        raise ValueError("outputs.root is required")


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    project_root: str | Path | None = None,
) -> Config:
    """Load default YAML config, apply optional file and dotted-key overrides."""
    root = Path(project_root).resolve() if project_root is not None else find_project_root()
    default_path = Path(__file__).with_name("default.yaml")
    with default_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if path is not None:
        user_path = Path(path)
        user_path = user_path if user_path.is_absolute() else root / user_path
        with user_path.open("r", encoding="utf-8") as file:
            config = _deep_merge(config, yaml.safe_load(file) or {})

    config = apply_overrides(config, overrides)
    config = resolve_config_paths(config, root)
    validate_config(config)
    return config


def parse_cli_overrides(values: list[str] | None) -> dict[str, Any]:
    """Parse ``key=value`` CLI override strings."""
    parsed: dict[str, Any] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"Override must use key=value syntax: {item!r}")
        key, value = item.split("=", 1)
        parsed[key] = value
    return parsed
