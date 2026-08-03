#!/usr/bin/env python3
"""
scripts/evaluate.py
--------------------
Generates the self-contained evaluation report required by NF5:
Accuracy, Precision, Recall, Macro-F1, False Positive Rate, and a
full confusion matrix for the hybrid system — plus the NF1 empirical
latency benchmark for the full pipeline (three models + fuzzy
aggregator) on live-style single-record inference.

Can be run standalone against a saved model + a held-out CSV, or
imported and called directly by scripts/train.py right after training.

Usage:
    python scripts/evaluate.py --data data/KDDTest+.csv
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)

from config import CLASS_LABELS, LOG_DIR  # noqa: E402
from pipeline import NIDSPipeline  # noqa: E402


def _false_positive_rate(cm: np.ndarray) -> dict:
    """Per-class FPR = FP / (FP + TN), computed one-vs-rest from the
    confusion matrix, plus a macro-average across classes."""
    n = cm.shape[0]
    fpr = {}
    for i in range(n):
        fp = cm[:, i].sum() - cm[i, i]
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fpr[CLASS_LABELS[i]] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fpr["macro_avg"] = float(np.mean(list(fpr.values())))
    return fpr


def run_evaluation(pipeline: NIDSPipeline, X_test: np.ndarray, y_test: np.ndarray,
                    out_path: str = None) -> dict:
    svm_p = pipeline.svm.predict_proba(X_test)
    rf_p = pipeline.rf.predict_proba(X_test)
    lstm_p = pipeline.lstm.predict_proba(X_test)

    from src.fuzzy_aggregator import FuzzyAggregator
    fusion = FuzzyAggregator.build_fusion_vector(svm_p, rf_p, lstm_p)
    result = pipeline.fuzzy.predict(fusion)
    y_pred = result.predicted_class_idx

    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(CLASS_LABELS))))

    report = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "false_positive_rate": _false_positive_rate(cm),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": CLASS_LABELS,
        "classification_report": classification_report(
            y_test, y_pred, labels=list(range(len(CLASS_LABELS))),
            target_names=CLASS_LABELS, zero_division=0, output_dict=True,
        ),
        "analyst_review_queue_rate": float(np.mean(result.needs_review)),
        "latency_benchmark_ms": _latency_benchmark(pipeline, X_test),
    }

    out_path = out_path or os.path.join(LOG_DIR, "evaluation_report.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n=== AI-NIDS Evaluation Report ===")
    print(f"Accuracy:        {report['accuracy']:.4f}")
    print(f"Precision(macro):{report['precision_macro']:.4f}")
    print(f"Recall(macro):   {report['recall_macro']:.4f}")
    print(f"F1(macro):       {report['f1_macro']:.4f}")
    print(f"FPR(macro):      {report['false_positive_rate']['macro_avg']:.4f}")
    print(f"Review-queue rate: {report['analyst_review_queue_rate']:.4f}")
    print(f"Mean pipeline latency: {report['latency_benchmark_ms']['mean_total_ms']:.3f} ms/record")
    print(f"Confusion matrix ({CLASS_LABELS}):")
    for row in cm:
        print("  ", row.tolist())
    print(f"\nFull report written to {out_path}")

    return report


def _latency_benchmark(pipeline: NIDSPipeline, X_test: np.ndarray, n_samples: int = 200) -> dict:
    """NF1: empirically measures single-record inference latency for the
    full hybrid pipeline (three models + fuzzy aggregator), simulating
    live-traffic, one-record-at-a-time conditions rather than batch
    throughput, which would understate real-world per-packet latency."""
    n_samples = min(n_samples, X_test.shape[0])
    idx = np.random.RandomState(42).choice(X_test.shape[0], n_samples, replace=False)

    totals, svm_lats, rf_lats, lstm_lats = [], [], [], []
    from src.fuzzy_aggregator import FuzzyAggregator

    for i in idx:
        record = X_test[i:i + 1]
        start = time.perf_counter()
        svm_p, svm_lat = pipeline.svm.timed_predict_proba(record)
        rf_p, rf_lat = pipeline.rf.timed_predict_proba(record)
        lstm_p, lstm_lat = pipeline.lstm.timed_predict_proba(record)
        fusion = FuzzyAggregator.build_fusion_vector(svm_p, rf_p, lstm_p)
        pipeline.fuzzy.predict(fusion)
        totals.append((time.perf_counter() - start) * 1000)
        svm_lats.append(svm_lat * 1000)
        rf_lats.append(rf_lat * 1000)
        lstm_lats.append(lstm_lat * 1000)

    return {
        "n_samples": n_samples,
        "mean_total_ms": float(np.mean(totals)),
        "p95_total_ms": float(np.percentile(totals, 95)),
        "mean_svm_ms": float(np.mean(svm_lats)),
        "mean_random_forest_ms": float(np.mean(rf_lats)),
        "mean_lstm_ms": float(np.mean(lstm_lats)),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained AI-NIDS pipeline.")
    parser.add_argument("--data", required=True, help="Held-out labelled CSV (e.g. KDDTest+.csv)")
    args = parser.parse_args()

    pipeline = NIDSPipeline().load()
    df, _ = pipeline.ingestor.load_csv(args.data)
    X, y = pipeline.preprocessor.transform(df)
    X_df = pipeline.preprocessor.transform_dataframe(X)
    X_sel = pipeline.feature_extractor.transform(X_df).to_numpy()

    run_evaluation(pipeline, X_sel, y)


if __name__ == "__main__":
    main()
