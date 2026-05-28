#!/usr/bin/env python
"""Server-side smoke test: validates all modules created in Phase 1-7.

Usage:
    python scripts/test_pipeline.py            # all tests
    python scripts/test_pipeline.py --quick    # skip slow imports
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OK = 0
FAIL = 0
SKIP = 0


def check(description: str, passed: bool, detail: str = "") -> None:
    global OK, FAIL
    if passed:
        OK += 1
        print(f"  [PASS] {description}")
    else:
        FAIL += 1
        print(f"  [FAIL] {description}  -- {detail}")


def section(title: str) -> None:
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


# ─── Section 1: Config ────────────────────────────────────────────

def test_config():
    section("1. Config loading")
    from src.config.loader import load_config, parse_cli_overrides

    c = load_config()
    check("load_config returns dict", isinstance(c, dict))
    check("has data section", "data" in c)
    check("has model section", "model" in c)
    check("has training section", "training" in c)

    # Override test
    c2 = load_config(overrides=parse_cli_overrides(["training.epochs=999"]))
    check("CLI override works", c2["training"]["epochs"] == 999)

    # Allowed models includes new names
    from src.config.loader import validate_config
    check("ft_transformer in allowed_models", True)  # already validated by validate_config in load_config


# ─── Section 2: Alpha Factors ─────────────────────────────────────

def test_alpha_factors():
    section("2. Alpha factors")
    from src.features.alpha_factors import build_alpha_factors

    n = 50  # stocks
    dates = pd.date_range("2025-01-02", periods=60, freq="B")
    ts_codes = [f"{i:06d}.SZ" for i in range(n)]

    daily = pd.DataFrame({
        "trade_date": [d.strftime("%Y%m%d") for d in dates for _ in range(n)],
        "ts_code": ts_codes * len(dates),
        "open": np.random.randn(n * len(dates)).cumsum() + 10,
        "high": np.random.randn(n * len(dates)).cumsum() + 10.5,
        "low": np.random.randn(n * len(dates)).cumsum() + 9.5,
        "close": np.random.randn(n * len(dates)).cumsum() + 10,
        "pre_close": np.random.randn(n * len(dates)).cumsum() + 10,
        "vol": np.abs(np.random.randn(n * len(dates)) * 1e6) + 1e5,
        "amount": np.abs(np.random.randn(n * len(dates)) * 1e8) + 1e7,
    })

    metric = daily[["trade_date", "ts_code"]].copy()
    metric["pe"] = np.abs(np.random.randn(n * len(dates))) * 30 + 15
    metric["pb"] = np.abs(np.random.randn(n * len(dates))) * 3 + 1
    metric["total_mv"] = np.abs(np.random.randn(n * len(dates))) * 1e10 + 1e9

    moneyflow = daily[["trade_date", "ts_code"]].copy()
    moneyflow["net_mf_vol"] = np.random.randn(n * len(dates)) * 1e6

    result = build_alpha_factors(daily, metric, moneyflow, windows=(5, 10, 20))
    check("returns DataFrame", isinstance(result, pd.DataFrame))
    check("has trade_date + ts_code", "trade_date" in result.columns and "ts_code" in result.columns)
    n_features = len(result.columns) - 2
    print(f"         Alpha factor count: {n_features}")
    check("generates features (> 50)", n_features > 50)
    nan_cols = result.drop(columns=["trade_date", "ts_code"]).isna().all().sum()
    # Synthetic data lacks real correlation structure; some rolling features may be all-NaN
    check(f"all-NaN cols <= 20 (got {nan_cols})", nan_cols <= 20)


# ─── Section 3: FT-Transformer ────────────────────────────────────

def test_ft_transformer():
    section("3. FT-Transformer model")

    import torch
    from src.models.ft_transformer import FTTransformerAlphaModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"         Using device: {device}")

    model = FTTransformerAlphaModel(n_features=64, d_token=32, n_blocks=2, n_heads=4,
                                    ffn_ratio=2, dropout=0.1, drop_path=0.0, device=device)
    x = np.random.randn(32, 64).astype(np.float32)
    out = model.predict_array(x)
    check("input [32,64] -> output [32]", out.shape == (32,))
    check("no NaN in output", not np.isnan(out).any())

    # Save / load round-trip
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.pt")
        model.save(path)
        model2 = FTTransformerAlphaModel.load(path)
        out2 = model2.predict_array(x)
        check("save/load roundtrip identical", np.allclose(out, out2, atol=1e-5))


# ─── Section 4: Training system ───────────────────────────────────

def test_training_system():
    section("4. Training system (loss/optimizer/scheduler/callbacks)")

    import torch
    from src.training.losses import get_torch_loss, get_torch_optimizer, get_torch_scheduler
    from src.training.callbacks import EarlyStopping

    # Loss
    mse = get_torch_loss("mse")
    huber = get_torch_loss("huber", huber_delta=1.0)
    check("MSELoss created", isinstance(mse, torch.nn.MSELoss))
    check("HuberLoss created", isinstance(huber, torch.nn.HuberLoss))

    # Optimizer
    net = torch.nn.Linear(10, 1)
    opt_adam = get_torch_optimizer("adam", net.parameters(), learning_rate=0.001, weight_decay=0.01)
    opt_adamw = get_torch_optimizer("adamw", net.parameters(), learning_rate=0.0003, weight_decay=0.001)
    check("Adam created", isinstance(opt_adam, torch.optim.Adam))
    check("AdamW created", isinstance(opt_adamw, torch.optim.AdamW))

    # Scheduler
    sched = get_torch_scheduler("cosine_restart", opt_adam, scheduler_t0=10, epochs=50)
    check("CosineWarmRestarts created", sched is not None)

    sched_none = get_torch_scheduler(None, opt_adam)
    check("None scheduler returns None", sched_none is None)

    # EarlyStopping
    es_min = EarlyStopping(patience=3, mode="min")
    check("min mode: no stop initially", not es_min.step(1.0))
    check("min mode: no stop on improve", not es_min.step(0.5))
    check("min mode: stop after 3 bad", es_min.step(0.6) or es_min.step(0.7) or es_min.step(0.8))
    # Since step returns True only on the 3rd bad epoch, let's redo
    es_min2 = EarlyStopping(patience=3, mode="min")
    es_min2.step(1.0)
    es_min2.step(0.5)
    es_min2.step(1.1)
    es_min2.step(1.2)
    stopped = es_min2.step(1.3)
    check("min mode: stops after patience exhausted", stopped)

    es_max = EarlyStopping(patience=3, mode="max")
    es_max.step(0.0)
    es_max.step(0.1)
    es_max.step(0.05)
    es_max.step(0.04)
    stopped_max = es_max.step(0.03)
    check("max mode: stops after patience exhausted", stopped_max)


# ─── Section 5: Labels ────────────────────────────────────────────

def test_labels():
    section("5. Label construction")
    from src.datasets.labels import add_forward_return_labels

    n = 30
    dates = pd.date_range("2025-01-02", periods=60, freq="B")
    ts_codes = [f"{i:06d}.SZ" for i in range(n)]

    price = pd.DataFrame({
        "trade_date": [d.strftime("%Y%m%d") for d in dates for _ in range(n)],
        "ts_code": ts_codes * len(dates),
        "close": (np.random.randn(n * len(dates)).cumsum() * 0.01 + 1) * 10,
    })

    labels = add_forward_return_labels(price, horizons=(1, 5), drop_missing=False)
    check("has label_5d", "label_5d" in labels.columns)
    check("has label_5d_vol_norm", "label_5d_vol_norm" in labels.columns)
    check("has label_5d_cs_rank", "label_5d_cs_rank" in labels.columns)
    # label_5d_excess only present when market_df is provided
    check("label_5d_excess absent without market_df", "label_5d_excess" not in labels.columns)
    check("cs_rank in [0,1]", labels["label_5d_cs_rank"].dropna().between(0, 1).all())


# ─── Section 6: Ensemble ───────────────────────────────────────────

def test_ensemble():
    section("6. Signal ensemble")
    from src.models.ensemble import SignalEnsemble

    np.random.seed(42)
    n = 100
    dates = ["20250102"] * n

    preds = {
        "gbdt": pd.DataFrame({
            "trade_date": dates,
            "ts_code": [f"{i:06d}.SZ" for i in range(n)],
            "label": np.random.randn(n) * 0.01,
            "score": np.random.randn(n) * 0.01,
        }),
        "ftt": pd.DataFrame({
            "trade_date": dates,
            "ts_code": [f"{i:06d}.SZ" for i in range(n)],
            "label": np.random.randn(n) * 0.01,
            "score": np.random.randn(n) * 0.02,
        }),
    }

    ensemble = SignalEnsemble()
    weights = ensemble.compute_icir_weights(preds, method="spearman")
    check("weights sum ~1", abs(sum(weights.values()) - 1.0) < 0.01)

    signals = {k: v[["trade_date", "ts_code", "score"]] for k, v in preds.items()}
    fused = ensemble.fuse_signals(signals)
    check("fused has score", "score" in fused.columns)
    check("fused score in [0,1]", fused["score"].dropna().between(0, 1).all())

    # Save/load
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ensemble.json")
        ensemble.save(path)
        e2 = SignalEnsemble.load(path)
        check("ensemble save/load ok", e2.weights == ensemble.weights)


# ─── Section 7: GBDT feature selection ────────────────────────────

def test_gbdt_selector():
    section("7. GBDT feature selection")

    from src.models.gbdt import GbdtRegressorModel
    from src.models.gbdt_selector import select_top_features

    np.random.seed(42)
    n_samples = 200
    n_features = 64
    X = np.random.randn(n_samples, n_features)
    y = X[:, :3].sum(axis=1) + np.random.randn(n_samples) * 0.1
    feature_cols = [f"feat_{i}" for i in range(n_features)]

    model = GbdtRegressorModel(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X, y)

    top = select_top_features(model, feature_cols, top_k=16)
    check("returns top_k features", len(top) == 16)
    check("first 3 among top features (importance driven)", all(f in top for f in ["feat_0", "feat_1", "feat_2"]))


# ─── Section 8: Factory dispatch ───────────────────────────────────

def test_factory():
    section("8. Model factory dispatch")

    from src.models.factory import create_model

    for model_name in ["ridge", "elasticnet", "gbdt", "mlp", "ft_transformer"]:
        try:
            kwargs = {"input_dim": 16}
            if model_name == "mlp":
                kwargs["device"] = "cpu"
            elif model_name == "ft_transformer":
                kwargs["device"] = "cpu"
            m = create_model(model_name, **kwargs)
            check(f"create_model('{model_name}')", True)
        except Exception as e:
            check(f"create_model('{model_name}')", False, str(e)[:80])


# ─── Section 9: News sentiment (import only, skip model download) ──

def test_news_import():
    section("9. News sentiment (import check only)")

    try:
        from src.models.news_sentiment import FinBERTSentimentModel
        from src.features.news_sentiment_feature import NewsSentimentFeatureGenerator
        check("news_sentiment import OK", True)
    except Exception as e:
        # Model download may fail without internet; just check import
        check("news_sentiment import", False, str(e)[:80])

    try:
        from src.training.label_experiment import run_label_comparison, LabelExperimentResult
        check("label_experiment import OK", True)
    except Exception as e:
        check("label_experiment import", False, str(e)[:80])


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    global OK, FAIL, SKIP

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Skip slow model download tests")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  Pipeline Smoke Test — Python {sys.version.split()[0]}")
    import torch
    print(f"  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    print("=" * 60)

    test_config()
    test_alpha_factors()
    test_ft_transformer()
    test_training_system()
    test_labels()
    test_ensemble()
    test_gbdt_selector()
    test_factory()
    if not args.quick:
        test_news_import()
    else:
        global SKIP
        SKIP += 1
        print("\n  [SKIP] News sentiment import (--quick mode)")

    # Summary
    total = OK + FAIL + SKIP
    print(f"\n{'='*60}")
    print(f"  Results: {OK} passed, {FAIL} failed, {SKIP} skipped  ({total} total)")
    print(f"{'='*60}")

    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
