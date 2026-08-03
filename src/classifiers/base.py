"""
base.py
-------
Common interface all Stage-4 ensemble members implement, so
FuzzyAggregator can treat SVM / RandomForest / LSTM interchangeably.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Optional

import joblib
import numpy as np


class BaseClassifier(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseClassifier":
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Must return an (n_samples, n_classes) array of class probabilities."""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def timed_predict_proba(self, X: np.ndarray) -> tuple:
        """Returns (probabilities, latency_seconds) for NF1 benchmarking."""
        start = time.perf_counter()
        proba = self.predict_proba(X)
        latency = time.perf_counter() - start
        return proba, latency

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "BaseClassifier":
        return joblib.load(path)
