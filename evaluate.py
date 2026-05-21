"""CLI entry point for prediction evaluation."""

from __future__ import annotations

import argparse

import pandas as pd

from src.evaluation.metrics import save_evaluation_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-column", default="label")
    args = parser.parse_args()
    predictions = pd.read_csv(args.predictions)
    save_evaluation_report(predictions, args.output_dir, args.label_column)


if __name__ == "__main__":
    main()

