"""
pipeline.py
-----------
NIDSPipeline: orchestrates the full five-stage flow end-to-end and is
the single object the Flask dashboard, training script, and live-mode
script all drive. Keeping orchestration here (rather than duplicating
it in scripts/) satisfies NF4 (maintainability) — there is exactly one
place that defines "what the pipeline does".
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from config import CLASS_LABELS, MODEL_DIR, TOP_K_FEATURES
from src.alert_manager import AlertManager
from src.classifiers import LSTMClassifier, RandomForestClassifier, SVMClassifier
from src.data_ingestor import DataIngestor
from src.feature_extractor import FeatureExtractor
from src.fuzzy_aggregator import FuzzyAggregator
from src.preprocessor import Preprocessor

logger = logging.getLogger("ai_nids.pipeline")


@dataclass
class InferenceOutcome:
    predicted_label: str
    confidence: float
    needs_review: bool
    per_model_latency_ms: dict = field(default_factory=dict)
    total_latency_ms: float = 0.0


class NIDSPipeline:
    """Wires DataIngestor -> Preprocessor -> FeatureExtractor ->
    {SVM, RandomForest, LSTM} -> FuzzyAggregator -> AlertManager."""

    def __init__(self, model_dir: str = MODEL_DIR, alert_manager: Optional[AlertManager] = None):
        self.model_dir = model_dir
        self.ingestor = DataIngestor()
        self.preprocessor = Preprocessor()
        self.feature_extractor = FeatureExtractor(top_k=TOP_K_FEATURES)
        self.svm = SVMClassifier()
        self.rf = RandomForestClassifier()
        self.lstm = LSTMClassifier()
        self.fuzzy = FuzzyAggregator()
        self.alert_manager = alert_manager or AlertManager()
        self._is_trained = False

    # ------------------------------------------------------------------
    # Training (FR5: supports full retraining on new balanced datasets)
    # ------------------------------------------------------------------
    def train(self, csv_path: str, test_size: float = 0.2) -> dict:
        from sklearn.model_selection import train_test_split

        df, ingest_report = self.ingestor.load_csv(csv_path)
        if df.empty:
            raise ValueError(f"No valid records ingested from {csv_path}")

        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=42,
            stratify=df["class"] if df["class"].nunique() > 1 else None,
        )

        # Stage 2: preprocessing + SMOTE (train only)
        X_train, y_train = self.preprocessor.fit_transform(train_df, balance=True)
        X_test, y_test = self.preprocessor.transform(test_df)

        # Stage 3: feature selection (fit on train, applied to both)
        X_train_df = self.preprocessor.transform_dataframe(X_train)
        X_test_df = self.preprocessor.transform_dataframe(X_test)
        X_train_sel = self.feature_extractor.fit_transform(X_train_df, y_train).to_numpy()
        X_test_sel = self.feature_extractor.transform(X_test_df).to_numpy()

        # Stage 4: train ensemble members in parallel (conceptually;
        # sequential here for reproducibility and simpler resource use
        # on a typical grading machine — see NF1 note in evaluate.py for
        # the true-parallel latency benchmark at inference time)
        logger.info("Training SVM (RBF kernel)...")
        self.svm.fit(X_train_sel, y_train)

        logger.info("Training Random Forest (100 trees)...")
        self.rf.fit(X_train_sel, y_train)

        logger.info("Training LSTM (sequence_window=%d)...",
                    self.lstm.params["sequence_window"])
        self.lstm.fit(X_train_sel, y_train)

        # Stage 5: fit + calibrate fuzzy aggregator on the training split
        svm_p = self.svm.predict_proba(X_train_sel)
        rf_p = self.rf.predict_proba(X_train_sel)
        lstm_p = self.lstm.predict_proba(X_train_sel)
        fusion_train = FuzzyAggregator.build_fusion_vector(svm_p, rf_p, lstm_p)
        self.fuzzy.fit(fusion_train)
        self.fuzzy.calibrate(fusion_train, y_train)

        self._is_trained = True
        self.save()

        self.alert_manager.record_audit_event(
            "MODEL_RETRAINED",
            f"dataset={csv_path} train_rows={len(train_df)} test_rows={len(test_df)} "
            f"ingest_report={json.dumps(ingest_report.as_dict())}",
        )

        return {
            "ingest_report": ingest_report.as_dict(),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "test_data": (X_test_sel, y_test),  # returned for evaluate.py
        }

    # ------------------------------------------------------------------
    # Inference (single record or small batch — used by both the
    # dashboard's "classify" action and run_live.py)
    # ------------------------------------------------------------------
    def infer(self, df: pd.DataFrame) -> List[InferenceOutcome]:
        if not self._is_trained:
            raise RuntimeError("Pipeline has not been trained/loaded. Call train() or load().")

        X, _ = self.preprocessor.transform(df)
        X_df = self.preprocessor.transform_dataframe(X)
        X_sel = self.feature_extractor.transform(X_df).to_numpy()

        total_start = time.perf_counter()

        svm_p, svm_lat = self.svm.timed_predict_proba(X_sel)
        rf_p, rf_lat = self.rf.timed_predict_proba(X_sel)
        lstm_p, lstm_lat = self.lstm.timed_predict_proba(X_sel)

        fusion = FuzzyAggregator.build_fusion_vector(svm_p, rf_p, lstm_p)
        result = self.fuzzy.predict(fusion)

        total_latency_ms = (time.perf_counter() - total_start) * 1000
        per_model_latency = {
            "svm_ms": svm_lat * 1000, "random_forest_ms": rf_lat * 1000,
            "lstm_ms": lstm_lat * 1000,
        }

        outcomes = []
        for i in range(len(df)):
            outcomes.append(InferenceOutcome(
                predicted_label=result.predicted_class_label[i],
                confidence=float(result.max_membership[i]),
                needs_review=bool(result.needs_review[i]),
                per_model_latency_ms=per_model_latency,
                total_latency_ms=total_latency_ms / len(df),
            ))
        return outcomes

    def infer_and_alert(self, df: pd.DataFrame) -> List[Optional[int]]:
        """Runs infer() and raises an AlertManager alert for every record
        whose predicted class is not 'Normal' (FR4), routing low-confidence
        predictions to the review queue instead of an immediate alert."""
        outcomes = self.infer(df)
        alert_ids = []
        for (_, row), outcome in zip(df.iterrows(), outcomes):
            if outcome.predicted_label == "Normal":
                alert_ids.append(None)
                continue
            alert_id = self.alert_manager.raise_alert(
                attack_category=outcome.predicted_label,
                confidence=outcome.confidence,
                src_ip=str(row.get("src_ip", "unknown")),
                dst_ip=str(row.get("dst_ip", "unknown")),
                src_port=int(row.get("src_port", 0) or 0),
                dst_port=int(row.get("dst_port", 0) or 0),
                needs_review=outcome.needs_review,
            )
            alert_ids.append(alert_id)
        return alert_ids

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self) -> None:
        os.makedirs(self.model_dir, exist_ok=True)
        import joblib
        self.svm.save(os.path.join(self.model_dir, "svm.joblib"))
        self.rf.save(os.path.join(self.model_dir, "rf.joblib"))
        self.lstm.save(os.path.join(self.model_dir, "lstm"))
        joblib.dump(self.preprocessor, os.path.join(self.model_dir, "preprocessor.joblib"))
        joblib.dump(self.feature_extractor, os.path.join(self.model_dir, "feature_extractor.joblib"))
        joblib.dump(self.fuzzy, os.path.join(self.model_dir, "fuzzy.joblib"))
        logger.info("Pipeline artifacts saved to %s", self.model_dir)

    def load(self) -> "NIDSPipeline":
        import joblib
        self.svm = SVMClassifier.load(os.path.join(self.model_dir, "svm.joblib"))
        self.rf = RandomForestClassifier.load(os.path.join(self.model_dir, "rf.joblib"))
        self.lstm = LSTMClassifier.load(os.path.join(self.model_dir, "lstm"))
        self.preprocessor = joblib.load(os.path.join(self.model_dir, "preprocessor.joblib"))
        self.feature_extractor = joblib.load(os.path.join(self.model_dir, "feature_extractor.joblib"))
        self.fuzzy = joblib.load(os.path.join(self.model_dir, "fuzzy.joblib"))
        self._is_trained = True
        return self
