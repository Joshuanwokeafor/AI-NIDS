"""rf_classifier.py — 100-tree Random Forest ensemble member (Stage 4)."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier as _SKRandomForest

from config import RF_PARAMS
from .base import BaseClassifier


class RandomForestClassifier(BaseClassifier):
    name = "random_forest"

    def __init__(self, **overrides):
        params = {**RF_PARAMS, **overrides}
        self.model = _SKRandomForest(**params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_
