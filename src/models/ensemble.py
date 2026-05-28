"""Multi-signal ensemble via ICIR-weighted rank fusion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.schema import TRADE_DATE, TS_CODE
from src.evaluation.metrics import daily_ic


class SignalEnsemble:
    """Fuses predictions from multiple models using ICIR-weighted rank fusion.

    Fusion is performed at the cross-sectional rank percentile level to handle
    scale differences between models (GBDT outputs in return space, FT-Transformer
    in learned space, News sentiment in [-1, 1]).
    """

    def __init__(self):
        self.weights: dict[str, float] = {}
        self._icir: dict[str, float] = {}

    def compute_icir_weights(
        self,
        valid_predictions: dict[str, pd.DataFrame],
        label_column: str = "label",
        method: str = "spearman",
    ) -> dict[str, float]:
        """Compute ICIR-based ensemble weights from validation predictions.

        Args:
            valid_predictions: Dict of model_name -> prediction DataFrame.
                Each DataFrame must have [trade_date, ts_code, label, score].
            label_column: Name of the label column.
            method: "pearson" for IC or "spearman" for Rank IC.

        Returns:
            Dict of model_name -> weight (sums to 1).
        """
        icir_values: dict[str, float] = {}
        for name, preds in valid_predictions.items():
            ic_frame = daily_ic(preds, label_column=label_column, method=method)
            ic_mean = ic_frame["ic"].mean()
            ic_std = ic_frame["ic"].std()
            icir = ic_mean / ic_std if ic_std > 0 else 0.0
            icir_values[name] = icir

        self._icir = icir_values
        # Only keep positive ICIR models; zero out negative ones
        positive_icir = {k: max(0.0, v) for k, v in icir_values.items()}
        total = sum(positive_icir.values())
        if total > 0:
            self.weights = {k: v / total for k, v in positive_icir.items()}
        else:
            # Fallback: equal weight
            n = len(positive_icir)
            self.weights = {k: 1.0 / n for k in positive_icir}

        return dict(self.weights)

    def fuse_signals(
        self,
        signals_dict: dict[str, pd.DataFrame],
        weights: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """Fuse multiple model signals into one ensemble score.

        Args:
            signals_dict: Dict of model_name -> signal DataFrame.
                Each DataFrame must have [trade_date, ts_code, score].
            weights: Optional override weights. Uses self.weights if None.

        Returns:
            DataFrame with [trade_date, ts_code, score, ensemble_weight_info].
        """
        if weights is None:
            weights = self.weights

        # Normalize each model's scores to cross-sectional rank percentiles
        rank_pcts: dict[str, pd.Series] = {}
        for name, signal in signals_dict.items():
            if name not in weights or weights[name] <= 0:
                continue
            rank_pcts[name] = signal.groupby(TRADE_DATE)["score"].rank(pct=True)

        # Fused score = weighted sum of rank percentiles
        fused_scores = pd.Series(0.0, index=next(iter(rank_pcts.values())).index)
        active_weights: dict[str, float] = {}

        for name, rank_pct in rank_pcts.items():
            active_weights[name] = weights[name]
            fused_scores = fused_scores.add(rank_pct * weights[name], fill_value=0)

        result = signals_dict[list(signals_dict.keys())[0]][[TRADE_DATE, TS_CODE]].copy()
        result["score"] = fused_scores.values
        result["ensemble_weight_info"] = json.dumps({
            "weights": {k: round(v, 4) for k, v in active_weights.items()},
            "icir": {k: round(v, 4) for k, v in self._icir.items()},
        })
        return result

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"weights": self.weights, "icir": self._icir}, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "SignalEnsemble":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        instance = cls()
        instance.weights = data.get("weights", {})
        instance._icir = data.get("icir", {})
        return instance


def compute_ensemble_weights_from_dir(
    predictions_dir: str | Path,
    model_names: list[str],
    label_column: str = "label",
) -> SignalEnsemble:
    """Load model predictions from a directory and compute ensemble weights."""
    ensemble = SignalEnsemble()
    preds: dict[str, pd.DataFrame] = {}
    for name in model_names:
        csv_path = Path(predictions_dir) / name / "valid_predictions.csv"
        if csv_path.exists():
            preds[name] = pd.read_csv(csv_path)
    ensemble.compute_icir_weights(preds, label_column=label_column)
    return ensemble
