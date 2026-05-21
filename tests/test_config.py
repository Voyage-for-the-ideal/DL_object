from __future__ import annotations

from pathlib import Path

from src.config.loader import load_config
from src.utils.io import ensure_output_tree
from src.utils.seed import set_global_seed


def test_default_config_loads_and_paths_resolve() -> None:
    config = load_config(project_root=Path.cwd())
    assert Path(config["data"]["root"]).is_absolute()
    assert config["training"]["seed"] == 42


def test_output_tree_and_seed_are_repeatable(tmp_path: Path) -> None:
    tree = ensure_output_tree(tmp_path)
    again = ensure_output_tree(tmp_path)
    assert set(tree) == set(again)
    set_global_seed(42)
