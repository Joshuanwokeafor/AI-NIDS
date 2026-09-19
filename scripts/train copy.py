#!/usr/bin/env python3
"""
scripts/train.py
-----------------
Trains the full AI-NIDS ensemble on an NSL-KDD CSV file and saves all
artifacts to models/. Also immediately runs evaluate.py's report
generation on the held-out split so a single command produces both a
trained system and the NF5 evaluation report for the project write-up.

Usage:
    python scripts/train.py --data data/KDDTrain+.csv
    python scripts/train.py --data data/KDDTrain+.csv --test-size 0.25
"""

import os

# MUST run before numpy/scipy/scikit-learn are imported anywhere (directly
# or transitively). OpenBLAS's runtime CPU dispatch can still select an
# AVX-family kernel on CPUs that report no AVX support at all, crashing
# with "Illegal instruction" on larger matrix operations even though small
# ones succeed. Forcing the SSE4.2-safe "Nehalem" kernel here fixes that
# on affected hosts and is a harmless no-op on hosts that don't need it.
# Override by exporting OPENBLAS_CORETYPE yourself before running this
# script if your CPU needs a different target.
os.environ.setdefault("OPENBLAS_CORETYPE", "Nehalem")

import argparse
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import NIDSPipeline  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Train the AI-NIDS ensemble.")
    parser.add_argument("--data", required=True, help="Path to NSL-KDD formatted CSV")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--report-out", default="logs/training_report.json")
    args = parser.parse_args()

    pipeline = NIDSPipeline()
    print(f"[*] Training on {args.data} (test_size={args.test_size}) ...")
    result = pipeline.train(args.data, test_size=args.test_size)

    X_test, y_test = result.pop("test_data")
    print(f"[*] Ingestion: {result['ingest_report']}")
    print(f"[*] Train rows: {result['train_rows']}  Test rows: {result['test_rows']}")
    print(f"[*] Model artifacts saved to {pipeline.model_dir}")

    os.makedirs(os.path.dirname(args.report_out), exist_ok=True)
    with open(args.report_out, "w") as f:
        json.dump(result, f, indent=2)

    print("[*] Running evaluation on held-out test split ...")
    from scripts.evaluate import run_evaluation
    run_evaluation(pipeline, X_test, y_test)


if __name__ == "__main__":
    main()
