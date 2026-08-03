"""
fuzzy_aggregator.py
--------------------
Stage 5 of the pipeline: FuzzyAggregator.

Concatenates the three ensemble members' probability vectors
(5 classes x 3 models = 15-dim fusion vector), applies Fuzzy c-means
clustering (5 clusters, matching the 5 output classes) to obtain a
continuous membership-degree vector per record, and applies
conditional threshold routing: records whose maximum cluster
membership falls below `delta` are routed to the analyst review queue
instead of producing an immediate automated alert.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import skfuzzy as fuzz

from config import (
    AMBIGUITY_THRESHOLD_DELTA, CLASS_LABELS, FUZZY_ERROR, FUZZY_M,
    FUZZY_MAX_ITER, FUZZY_N_CLUSTERS,
)

logger = logging.getLogger("ai_nids.fuzzy_aggregator")


@dataclass
class AggregationResult:
    fusion_vector: np.ndarray          # (n_samples, 15)
    membership: np.ndarray             # (n_samples, n_clusters)
    predicted_class_idx: np.ndarray    # (n_samples,) argmax of membership
    predicted_class_label: List[str]
    max_membership: np.ndarray         # (n_samples,)
    needs_review: np.ndarray           # bool (n_samples,) -> below delta


class FuzzyAggregator:
    """
    Fuses SVM / RandomForest / LSTM probability outputs via Fuzzy c-means
    and applies ambiguity-threshold routing (delta).
    """

    def __init__(
        self,
        n_clusters: int = FUZZY_N_CLUSTERS,
        m: float = FUZZY_M,
        error: float = FUZZY_ERROR,
        max_iter: int = FUZZY_MAX_ITER,
        delta: float = AMBIGUITY_THRESHOLD_DELTA,
    ):
        self.n_clusters = n_clusters
        self.m = m
        self.error = error
        self.max_iter = max_iter
        self.delta = delta
        self.cntr_: Optional[np.ndarray] = None  # fitted cluster centers
        self.cluster_to_class_: Optional[np.ndarray] = None  # calibrated mapping
        self._fitted = False

    # ------------------------------------------------------------------
    @staticmethod
    def build_fusion_vector(svm_proba: np.ndarray, rf_proba: np.ndarray,
                             lstm_proba: np.ndarray) -> np.ndarray:
        for name, arr in (("svm", svm_proba), ("random_forest", rf_proba),
                           ("lstm", lstm_proba)):
            if arr.ndim != 2:
                raise ValueError(f"{name}_proba must be 2D (n_samples, n_classes)")
        n = svm_proba.shape[0]
        if not (rf_proba.shape[0] == n and lstm_proba.shape[0] == n):
            raise ValueError("All three probability arrays must have the same n_samples")
        return np.hstack([svm_proba, rf_proba, lstm_proba])

    # ------------------------------------------------------------------
    def fit(self, fusion_vectors: np.ndarray) -> "FuzzyAggregator":
        if fusion_vectors.shape[0] < self.n_clusters:
            raise ValueError(
                f"Need at least n_clusters={self.n_clusters} samples to fit "
                f"fuzzy c-means, got {fusion_vectors.shape[0]}."
            )
        # skfuzzy expects features x samples
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            fusion_vectors.T, c=self.n_clusters, m=self.m,
            error=self.error, maxiter=self.max_iter, seed=42,
        )
        self.cntr_ = cntr
        self._fpc = fpc  # fuzzy partition coefficient, a cluster-quality metric
        self._fitted = True
        logger.info("FuzzyAggregator fitted: n_clusters=%d, fuzzy_partition_coeff=%.4f",
                    self.n_clusters, fpc)
        return self

    def predict(self, fusion_vectors: np.ndarray) -> AggregationResult:
        if not self._fitted:
            raise RuntimeError("FuzzyAggregator.fit() must be called before predict().")

        u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
            fusion_vectors.T, self.cntr_, m=self.m, error=self.error,
            maxiter=self.max_iter, seed=42,
        )
        membership = u.T  # (n_samples, n_clusters)
        predicted_idx = np.argmax(membership, axis=1)
        max_membership = membership[np.arange(len(membership)), predicted_idx]
        needs_review = max_membership < self.delta

        predicted_idx = self._align_clusters_to_classes(predicted_idx)
        labels = [CLASS_LABELS[i] for i in predicted_idx]

        return AggregationResult(
            fusion_vector=fusion_vectors,
            membership=membership,
            predicted_class_idx=predicted_idx,
            predicted_class_label=labels,
            max_membership=max_membership,
            needs_review=needs_review,
        )

    def fit_predict(self, fusion_vectors: np.ndarray) -> AggregationResult:
        self.fit(fusion_vectors)
        return self.predict(fusion_vectors)

    # ------------------------------------------------------------------
    def calibrate(self, fusion_vectors: np.ndarray, y_true: np.ndarray) -> "FuzzyAggregator":
        """
        Fuzzy c-means clusters are unordered by construction, so cluster #k
        has no inherent relationship to class label #k. This method learns
        that mapping via majority vote: for each cluster, assign it the
        true class label that appears most often among the (validation)
        records whose highest membership falls in that cluster. Must be
        called once, on a held-out labelled batch, after fit() and before
        predict() is used for anything other than raw membership scores.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before calibrate().")

        u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
            fusion_vectors.T, self.cntr_, m=self.m, error=self.error,
            maxiter=self.max_iter, seed=42,
        )
        raw_cluster_idx = np.argmax(u.T, axis=1)

        mapping = np.zeros(self.n_clusters, dtype=int)
        for cluster_id in range(self.n_clusters):
            mask = raw_cluster_idx == cluster_id
            if not mask.any():
                mapping[cluster_id] = cluster_id  # no data fell here; leave as-is
                continue
            votes = np.bincount(y_true[mask], minlength=len(CLASS_LABELS))
            mapping[cluster_id] = int(np.argmax(votes))

        self.cluster_to_class_ = mapping
        logger.info("FuzzyAggregator calibrated cluster->class mapping: %s",
                    dict(enumerate(mapping.tolist())))
        return self

    def _align_clusters_to_classes(self, cluster_idx: np.ndarray) -> np.ndarray:
        if self.cluster_to_class_ is None:
            logger.warning(
                "FuzzyAggregator.calibrate() was never called; using raw "
                "cluster indices as class indices. Call calibrate() on a "
                "held-out labelled batch before trusting predicted labels."
            )
            return cluster_idx
        return self.cluster_to_class_[cluster_idx]
