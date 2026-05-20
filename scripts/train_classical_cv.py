#!/usr/bin/env python3
"""Train a classical foul classifier from extracted CV descriptors."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cv_foul_detection.classical_model import FEATURE_SETS, train_random_forest
from cv_foul_detection.io import write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a RandomForest baseline over classical CV features.")
    parser.add_argument("--features", required=True, help="Feature root produced by scripts/extract_cv_features.py.")
    parser.add_argument("--dataset", required=True, help="Dataset root containing split annotations.")
    parser.add_argument("--target", default="offence_severity", choices=["action", "offence", "severity", "offence_severity"])
    parser.add_argument("--feature-set", default="core", choices=sorted(FEATURE_SETS), help="Descriptor group used by the classifier.")
    parser.add_argument("--train-split", default="Train")
    parser.add_argument("--eval-split", default="Valid")
    parser.add_argument("--output", default="outputs/classical_cv")
    args = parser.parse_args()

    model, report = train_random_forest(
        args.features,
        args.dataset,
        target=args.target,
        train_split=args.train_split,
        eval_split=args.eval_split,
        feature_set=args.feature_set,
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / f"{args.target}_{args.feature_set}_random_forest.pkl"
    report_path = output / f"{args.target}_{args.feature_set}_report.json"
    with model_path.open("wb") as f:
        pickle.dump(model, f)
    write_json(report_path, report)
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
