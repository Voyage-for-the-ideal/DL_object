"""GBDT-based feature selection for FT-Transformer."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.datasets.tabular_dataset import select_feature_columns
from src.models.gbdt import GbdtRegressorModel
from src.training.trainer import train_tabular_model
from src.utils.io import ensure_dir


def select_top_features(
    gbdt_model: GbdtRegressorModel,
    feature_columns: list[str],
    top_k: int = 128,
    output_path: str | Path | None = None,
) -> list[str]:
    """Extract top-k features from a trained GBDT by importance score."""
    importance_df = gbdt_model.feature_importance(feature_columns)
    importance_df = importance_df.sort_values("importance", ascending=False)
    top_features = importance_df.head(top_k)["feature"].tolist()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        importance_df.head(top_k).to_csv(output_path, index=False)

    return top_features


def gbdt_select_and_train_ftt(
    config: dict[str, Any],
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    top_k: int = 128,
    output_dir: str | Path | None = None,
) -> tuple[list[str], dict[str, float]]:
    """Full GBDT feature selection pipeline:

    1. Train GBDT on all features
    2. Select top-k features by importance
    3. Train FT-Transformer on selected features
    4. Return feature list and FT-Transformer metrics
    """
    if output_dir is None:
        output_dir = Path(config["outputs"]["root"]) / "feature_selection"
    output_dir = ensure_dir(output_dir)
    label_col = config["label"]["main"]

    # Step 1: Train GBDT on all features
    print("[GBDT Selector] Step 1: Training GBDT on all features...")
    gbdt_config = dict(config)
    gbdt_config["model"] = dict(config.get("model", {}))
    gbdt_config["model"]["name"] = "gbdt"
    gbdt_dir = Path(output_dir) / "gbdt_full"
    gbdt_result = train_tabular_model(
        train_frame, valid_frame, gbdt_config, gbdt_dir, label_column=label_col
    )
    gbdt_metrics = gbdt_result.get("metrics", {})

    # 释放 GBDT 训练占用的 GPU 内存
    del gbdt_result
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()

    # Step 2: Load GBDT and extract importance
    print(f"[GBDT Selector] Step 2: Selecting top-{top_k} features...")
    gbdt_model = GbdtRegressorModel.load(gbdt_dir / "model.joblib")
    feature_columns = select_feature_columns(train_frame, label_col)
    top_features = select_top_features(
        gbdt_model, feature_columns, top_k=top_k,
        output_path=Path(output_dir) / "top_features.csv",
    )

    # Step 3: Subset frames to top-k features
    keep_cols = ["trade_date", "ts_code", label_col] + top_features
    train_subset = train_frame[[c for c in keep_cols if c in train_frame.columns]]
    valid_subset = valid_frame[[c for c in keep_cols if c in valid_frame.columns]]

    # Step 4: Train FT-Transformer on selected features
    print(f"[GBDT Selector] Step 3: Training FT-Transformer on {len(top_features)} features...")
    ftt_config = dict(config)
    ftt_config["model"] = dict(config.get("model", {}))
    ftt_config["model"]["name"] = "ft_transformer"
    ftt_dir = Path(output_dir) / "ft_transformer_selected"
    ftt_result = train_tabular_model(
        train_subset, valid_subset, ftt_config, ftt_dir, label_column=label_col
    )
    ftt_metrics = ftt_result.get("metrics", {})

    # Save summary
    summary = {
        "top_k": top_k,
        "gbdt_rank_ic": gbdt_metrics.get("rank_ic_mean"),
        "ftt_rank_ic": ftt_metrics.get("rank_ic_mean"),
        "top_features": top_features[:20],  # first 20 for display
    }
    with open(Path(output_dir) / "selection_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[GBDT Selector] Done. GBDT Rank IC={gbdt_metrics.get('rank_ic_mean'):.6f}, "
          f"FTT Rank IC={ftt_metrics.get('rank_ic_mean'):.6f}")
    return top_features, ftt_metrics
