"""svm_classifier.py — RBF-kernel SVM ensemble member (Stage 4)."""

from __future__ import annotations

import numpy as np
from sklearn.svm import SVC

from config import SVM_PARAMS
from .base import BaseClassifier


class SVMClassifier(BaseClassifier):
    name = "svm"

    def __init__(self, **overrides):
        params = {**SVM_PARAMS, **overrides}
        self.model = SVC(**params)
        self._n_classes: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMClassifier":
        self._n_classes = len(np.unique(y))
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
