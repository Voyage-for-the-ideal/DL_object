"""Prediction metrics for alpha scores."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.schema import TRADE_DATE
from src.utils.io import ensure_dir


def regression_metrics(frame: pd.DataFrame, label_column: str = "label") -> dict[str, float]:
    diff = pd.to_numeric(frame["score"], errors="coerce") - pd.to_numeric(
        frame[label_column], errors="coerce"
    )
    return {"mse": float(np.nanmean(diff**2)), "mae": float(np.nanmean(np.abs(diff)))}


def daily_ic(
    frame: pd.DataFrame,
    label_column: str = "label",
    method: str = "pearson",
    min_count: int = 3,
) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for trade_date, group in frame.groupby(TRADE_DATE):
        clean = group[["score", label_column]].dropna()
        if (
            len(clean) < min_count
            or clean["score"].nunique() < 2
            or clean[label_column].nunique() < 2
        ):
            continue
        left = clean["score"].rank() if method == "spearman" else clean["score"]
        right = clean[label_column].rank() if method == "spearman" else clean[label_column]
        rows.append(
            {
                TRADE_DATE: str(trade_date),
                "ic": float(left.corr(right)),
                "count": int(len(clean)),
            }
        )
    return pd.DataFrame(rows, columns=[TRADE_DATE, "ic", "count"])


def summarize_ic(ic_frame: pd.DataFrame, column: str = "ic") -> dict[str, float]:
    if ic_frame.empty:
        return {"mean": float("nan"), "std": float("nan"), "icir": float("nan")}
    values = pd.to_numeric(ic_frame[column], errors="coerce").dropna()
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    mean = float(values.mean())
    return {"mean": mean, "std": std, "icir": mean / std if std > 0 else float("nan")}


def direction_accuracy(frame: pd.DataFrame, label_column: str = "label") -> float:
    clean = frame[["score", label_column]].dropna()
    if clean.empty:
        return float("nan")
    return float((np.sign(clean["score"]) == np.sign(clean[label_column])).mean())


def group_returns(
    frame: pd.DataFrame,
    label_column: str = "label",
    n_groups: int = 5,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for trade_date, group in frame.groupby(TRADE_DATE):
        clean = group[["score", label_column]].dropna().copy()
        if len(clean) < n_groups:
            continue
        clean["group"] = pd.qcut(clean["score"].rank(method="first"), n_groups, labels=False) + 1
        grouped = clean.groupby("group")[label_column].mean()
        for group_id, value in grouped.items():
            rows.append(
                {TRADE_DATE: str(trade_date), "group": int(group_id), "mean_return": float(value)}
            )
    return pd.DataFrame(rows)


def evaluate_predictions(
    predictions: pd.DataFrame,
    label_column: str = "label",
    n_groups: int = 5,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    pearson = daily_ic(predictions, label_column=label_column, method="pearson")
    rank = daily_ic(predictions, label_column=label_column, method="spearman")
    pearson_summary = summarize_ic(pearson)
    rank_summary = summarize_ic(rank)
    metrics = regression_metrics(predictions, label_column=label_column)
    metrics.update(
        {
            "ic_mean": pearson_summary["mean"],
            "ic_std": pearson_summary["std"],
            "icir": pearson_summary["icir"],
            "rank_ic_mean": rank_summary["mean"],
            "rank_icir": rank_summary["icir"],
            "direction_accuracy": direction_accuracy(predictions, label_column),
        }
    )
    groups = group_returns(predictions, label_column=label_column, n_groups=n_groups)
    pearson = pearson.rename(columns={"ic": "pearson_ic"})
    rank = rank.rename(columns={"ic": "rank_ic"})
    daily = pearson[[TRADE_DATE, "pearson_ic", "count"]].merge(
        rank[[TRADE_DATE, "rank_ic"]],
        on=TRADE_DATE,
        how="outer",
    )
    return metrics, daily, groups


def save_evaluation_report(
    predictions: pd.DataFrame,
    output_dir: str | Path,
    label_column: str = "label",
) -> dict[str, float]:
    target = ensure_dir(output_dir)
    metrics, daily, groups = evaluate_predictions(predictions, label_column=label_column)
    daily.to_csv(target / "daily_ic.csv", index=False)
    groups.to_csv(target / "group_return.csv", index=False)
    with (target / "prediction_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    if not daily.empty:
        daily.plot(x=TRADE_DATE, y=[column for column in daily.columns if column.endswith("_ic")])
        plt.tight_layout()
        plt.savefig(target / "ic_curve.png")
        plt.close()
    if not groups.empty:
        groups.groupby("group")["mean_return"].mean().plot(kind="bar")
        plt.tight_layout()
        plt.savefig(target / "group_return_plot.png")
        plt.close()
    return metrics
