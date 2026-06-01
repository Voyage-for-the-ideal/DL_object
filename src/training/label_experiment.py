"""Label type comparison experiment."""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.training.trainer import train_tabular_model
from src.utils.io import ensure_dir


@dataclass
class LabelExperimentResult:
    model_name: str
    label_type: str
    valid_ic: float
    valid_rank_ic: float
    valid_icir: float | None
    metrics: dict[str, float]


def run_label_comparison(
    config: dict[str, Any],
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    label_types: list[str],
    model_names: list[str],
    output_dir: str | Path,
) -> list[LabelExperimentResult]:
    """Train each (model, label) pair and compare validation Rank IC."""
    output_dir = ensure_dir(output_dir)
    results: list[LabelExperimentResult] = []
    config = dict(config)

    # 将 PyTorch 模型排在前面，避免 GBDT 先占满 GPU 导致 PyTorch OOM
    _pytorch_models = {"ft_transformer", "mlp", "transformer_encoder"}
    ordered_models = sorted(model_names, key=lambda m: (m not in _pytorch_models, m))

    for label_type in label_types:
        for model_name in ordered_models:
            run_name = f"{model_name}_{label_type}"
            print(f"\n{'='*60}\n  Running: {run_name}\n{'='*60}")

            run_config = dict(config)
            run_config["label"] = dict(config.get("label", {}))
            run_config["label"]["main"] = label_type
            run_config["model"] = dict(config.get("model", {}))
            run_config["model"]["name"] = model_name

            run_dir = Path(output_dir) / run_name
            try:
                result_dict = train_tabular_model(
                    train_frame, valid_frame, run_config, run_dir, label_column=label_type
                )
                metrics = result_dict.get("metrics", {})
                result = LabelExperimentResult(
                    model_name=model_name,
                    label_type=label_type,
                    valid_ic=metrics.get("ic_mean", float("nan")),
                    valid_rank_ic=metrics.get("rank_ic_mean", float("nan")),
                    valid_icir=metrics.get("icir"),
                    metrics=metrics,
                )
            except Exception as e:
                print(f"  FAILED: {e}")
                result = LabelExperimentResult(
                    model_name=model_name,
                    label_type=label_type,
                    valid_ic=float("nan"),
                    valid_rank_ic=float("nan"),
                    valid_icir=None,
                    metrics={"error": str(e)},
                )
            results.append(result)

            # 释放 GPU 内存，防止后续训练 OOM
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()

    # Save comparison table
    table = pd.DataFrame([
        {"model": r.model_name, "label": r.label_type,
         "IC": round(r.valid_ic, 6), "Rank_IC": round(r.valid_rank_ic, 6)}
        for r in results
    ])
    table = table.sort_values("Rank_IC", ascending=False)
    table.to_csv(Path(output_dir) / "label_comparison.csv", index=False)

    summary = {
        "best_rank_ic": float(table["Rank_IC"].max()),
        "best_model_label": table.iloc[0].to_dict() if not table.empty else None,
        "results": [r.__dict__ for r in results],
    }
    with open(Path(output_dir) / "label_comparison.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(table.to_string(index=False))
    print(f"\nResults saved to {output_dir}")
    return results
