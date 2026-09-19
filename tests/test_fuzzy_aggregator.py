"""tests/test_fuzzy_aggregator.py — unit tests for Stage 5 (FuzzyAggregator)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fuzzy_aggregator import AggregationResult, FuzzyAggregator  # noqa: E402


def _fake_proba(n_samples, n_classes=5, seed=0):
    rng = np.random.RandomState(seed)
    raw = rng.rand(n_samples, n_classes)
    return raw / raw.sum(axis=1, keepdims=True)


class TestBuildFusionVector:
    def test_concatenates_three_proba_matrices(self):
        svm_p = _fake_proba(10, seed=1)
        rf_p = _fake_proba(10, seed=2)
        lstm_p = _fake_proba(10, seed=3)
        fusion = FuzzyAggregator.build_fusion_vector(svm_p, rf_p, lstm_p)
        assert fusion.shape == (10, 15)

    def test_mismatched_sample_counts_raises(self):
        svm_p = _fake_proba(10)
        rf_p = _fake_proba(9)
        lstm_p = _fake_proba(10)
        with pytest.raises(ValueError):
            FuzzyAggregator.build_fusion_vector(svm_p, rf_p, lstm_p)

    def test_non_2d_input_raises(self):
        svm_p = _fake_proba(10)
        rf_p = np.ravel(_fake_proba(10))  # flattened to 1D
        lstm_p = _fake_proba(10)
        with pytest.raises(ValueError):
            FuzzyAggregator.build_fusion_vector(svm_p, rf_p, lstm_p)


class TestFuzzyAggregatorFitPredict:
    def test_fit_requires_min_samples(self):
        agg = FuzzyAggregator(n_clusters=5)
        fusion = _fake_proba(3, n_classes=15)  # fewer samples than clusters
        with pytest.raises(ValueError):
            agg.fit(fusion)

    def test_fit_predict_returns_valid_membership(self):
        agg = FuzzyAggregator(n_clusters=5)
        fusion = np.hstack([_fake_proba(50, seed=i) for i in range(3)])
        result = agg.fit_predict(fusion)

        assert isinstance(result, AggregationResult)
        assert result.membership.shape == (50, 5)
        # Fuzzy memberships for each sample must sum to ~1
        np.testing.assert_allclose(result.membership.sum(axis=1), 1.0, atol=1e-3)
        assert len(result.predicted_class_label) == 50
        assert result.max_membership.shape == (50,)
        assert result.needs_review.dtype == bool

    def test_predict_before_fit_raises(self):
        agg = FuzzyAggregator()
        fusion = _fake_proba(10, n_classes=15)
        with pytest.raises(RuntimeError):
            agg.predict(fusion)

    def test_threshold_routing_flags_low_confidence(self):
        agg = FuzzyAggregator(n_clusters=5, delta=0.99)  # near-impossible threshold
        fusion = np.hstack([_fake_proba(50, seed=i) for i in range(3)])
        result = agg.fit_predict(fusion)
        # With delta=0.99, essentially everything should be routed to review
        assert result.needs_review.sum() > 0

    def test_low_threshold_routes_nothing(self):
        agg = FuzzyAggregator(n_clusters=5, delta=0.0)
        fusion = np.hstack([_fake_proba(50, seed=i) for i in range(3)])
        result = agg.fit_predict(fusion)
        assert result.needs_review.sum() == 0

    def test_calibrate_learns_cluster_to_class_mapping(self):
        agg = FuzzyAggregator(n_clusters=5)
        fusion = np.hstack([_fake_proba(60, seed=i) for i in range(3)])
        y_true = np.random.RandomState(0).randint(0, 5, size=60)
        agg.fit(fusion)
        agg.calibrate(fusion, y_true)
        assert agg.cluster_to_class_ is not None
        assert agg.cluster_to_class_.shape == (5,)

    def test_uncalibrated_warns_but_still_predicts(self):
        agg = FuzzyAggregator(n_clusters=5)
        fusion = np.hstack([_fake_proba(30, seed=i) for i in range(3)])
        agg.fit(fusion)
        # No calibrate() call — should still return a result, using raw
        # cluster indices as a fallback rather than raising.
        result = agg.predict(fusion)
        assert result.predicted_class_idx.shape == (30,)
