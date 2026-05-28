"""CLI entry point for tabular training from prepared panel CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config.loader import load_config, parse_cli_overrides
from src.training.trainer import train_tabular_model
from src.utils.io import make_run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--train-panel", required=True)
    parser.add_argument("--valid-panel", required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--label-experiment", action="store_true",
                        help="Run label type comparison instead of single model training")
    parser.add_argument("--label-types", nargs="+",
                        default=["label_5d", "label_5d_vol_norm", "label_5d_cs_rank"],
                        help="Label columns to compare")
    parser.add_argument("--label-models", nargs="+",
                        default=["gbdt", "ft_transformer"],
                        help="Models to test per label")
    args = parser.parse_args()
    overrides = parse_cli_overrides(args.override)
    if args.model_name:
        overrides["model.name"] = args.model_name
    config = load_config(args.config, overrides=overrides)

    if getattr(args, "label_experiment", False):
        from src.training.label_experiment import run_label_comparison
        train_df = pd.read_csv(args.train_panel)
        valid_df = pd.read_csv(args.valid_panel)
        results = run_label_comparison(
            config, train_df, valid_df,
            label_types=args.label_types,
            model_names=args.label_models,
            output_dir=config["outputs"]["root"] + "/label_experiment",
        )
        results.sort(key=lambda r: r.valid_rank_ic, reverse=True)
        print(f"\nBest: {results[0].model_name} x {results[0].label_type} "
              f"(Rank IC={results[0].valid_rank_ic:.6f})")
        return

    run_dir = make_run_dir(config["outputs"]["root"], args.run_id)
    train_frame = pd.read_csv(args.train_panel)
    valid_frame = pd.read_csv(args.valid_panel)
    label_column = config.get("label", {}).get("main", "label_5d")
    train_tabular_model(train_frame, valid_frame, config, run_dir, label_column=label_column)
    print(Path(run_dir))


if __name__ == "__main__":
    main()

