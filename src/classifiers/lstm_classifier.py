"""
lstm_classifier.py — LSTM ensemble member (Stage 4).

Captures temporal/sequential protocol dependencies by grouping
consecutive records into overlapping windows of length
`sequence_window` (default 10, config.LSTM_PARAMS) before feeding them
to the recurrent network. This directly addresses the "temporal
awareness" gap identified in the literature review, since a plain
feature vector treats each connection independently.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from config import LSTM_PARAMS
from .base import BaseClassifier

logger = logging.getLogger("ai_nids.lstm_classifier")


def _build_model(n_features: int, n_classes: int, params: dict):
    # Local import: keeps TensorFlow optional for callers that only need
    # SVM/RandomForest (e.g. fast unit tests, low-resource grading VMs).
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.Input(shape=(params["sequence_window"], n_features)),
        layers.LSTM(params["units"], dropout=params["dropout"],
                    recurrent_dropout=params["recurrent_dropout"],
                    return_sequences=False),
        layers.Dense(params["dense_units"], activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


class LSTMClassifier(BaseClassifier):
    name = "lstm"

    def __init__(self, **overrides):
        self.params = {**LSTM_PARAMS, **overrides}
        self.model = None
        self._n_classes = 0
        self._n_features = 0

    # ------------------------------------------------------------------
    def _to_sequences(self, X: np.ndarray, y: np.ndarray | None = None):
        """
        Builds one sliding window of length `sequence_window` ending at
        each input record (left-padded with repeats of the first row for
        the initial records that don't yet have `sequence_window - 1`
        predecessors). This guarantees output length == input length,
        which every other ensemble member and FuzzyAggregator assume, while
        still giving the LSTM recent-history context for every record —
        including the very first ones in a batch and single-record live
        inference.
        """
        w = self.params["sequence_window"]
        n = X.shape[0]

        pad = np.repeat(X[:1], w - 1, axis=0)
        X_padded = np.vstack([pad, X])  # length n + w - 1

        seqs = np.stack([X_padded[i:i + w] for i in range(n)])
        return seqs, y

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LSTMClassifier":
        self._n_classes = len(np.unique(y))
        self._n_features = X.shape[1]
        X_seq, y_seq = self._to_sequences(X, y)

        self.model = _build_model(self._n_features, self._n_classes, self.params)
        self.model.fit(
            X_seq, y_seq,
            batch_size=self.params["batch_size"],
            epochs=self.params["epochs"],
            verbose=0,
            validation_split=0.1 if len(X_seq) > 20 else 0.0,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LSTMClassifier.fit() must be called before predict_proba().")
        X_seq, _ = self._to_sequences(X)
        proba = self.model.predict(X_seq, verbose=0)
        return proba

    # ------------------------------------------------------------------
    # Keras models are not joblib-friendly; override save/load.
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path if path.endswith(".keras") else path + ".keras")
        meta_path = path + ".meta.npy"
        np.save(meta_path, {
            "params": self.params,
            "n_classes": self._n_classes,
            "n_features": self._n_features,
        }, allow_pickle=True)

    @classmethod
    def load(cls, path: str) -> "LSTMClassifier":
        import tensorflow as tf
        model_path = path if path.endswith(".keras") else path + ".keras"
        meta = np.load(path + ".meta.npy", allow_pickle=True).item()
        obj = cls(**meta["params"])
        obj.model = tf.keras.models.load_model(model_path)
        obj._n_classes = meta["n_classes"]
        obj._n_features = meta["n_features"]
        return obj
