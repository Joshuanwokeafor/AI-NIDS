"""
preprocessor.py
----------------
Stage 2 of the pipeline: Preprocessor.

2a. Normalization: Min-Max scaling of numeric features to [0, 1] via
    fit_transform, plus one-hot encoding of categorical attributes,
    expanding the feature space (41 base -> up to 122).
2b. SMOTE: applied ONLY to the training partition to correct class
    imbalance, targeting a 1:1 minority:majority ratio.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from config import (
    ATTACK_CATEGORY_MAP, CATEGORICAL_FEATURES, CLASS_TO_INDEX,
    NUMERIC_FEATURES, SMOTE_K_NEIGHBORS,
)

logger = logging.getLogger("ai_nids.preprocessor")


class Preprocessor:
    """
    Handles Stage 2 (Preprocessing and Balancing).

    Usage:
        pre = Preprocessor()
        X_train, y_train = pre.fit_transform(train_df, balance=True)
        X_test, y_test   = pre.transform(test_df)     # balance=False always for test
    """

    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self._numeric_cols = [c for c in NUMERIC_FEATURES]
        self._categorical_cols = list(CATEGORICAL_FEATURES)
        self._fitted = False
        self.output_feature_names_: Optional[list] = None

    # ------------------------------------------------------------------
    def _labels_to_indices(self, class_series: pd.Series) -> np.ndarray:
        unknown_labels = set()

        def map_one(raw_label: str) -> int:
            raw_label = str(raw_label).strip().lower().rstrip(".")
            category = ATTACK_CATEGORY_MAP.get(raw_label)
            if category is None:
                unknown_labels.add(raw_label)
                category = "DoS"  # conservative default: treat unseen attack
                                   # strings as the majority attack class
                                   # rather than silently as "Normal"
            return CLASS_TO_INDEX[category]

        mapped = class_series.apply(map_one).to_numpy()
        if unknown_labels:
            logger.warning("Encountered unseen attack labels, mapped to 'DoS': %s",
                            sorted(unknown_labels))
        return mapped

    # ------------------------------------------------------------------
    def fit_transform(
        self, df: pd.DataFrame, balance: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        if "class" not in df.columns:
            raise ValueError("Training dataframe must contain a 'class' column.")

        y = self._labels_to_indices(df["class"])

        num_df = df[self._numeric_cols].fillna(0.0)
        cat_df = df[self._categorical_cols].fillna("unknown").astype(str)

        X_num = self.scaler.fit_transform(num_df)
        X_cat = self.encoder.fit_transform(cat_df)

        self.output_feature_names_ = list(self._numeric_cols) + list(
            self.encoder.get_feature_names_out(self._categorical_cols)
        )
        X = np.hstack([X_num, X_cat])
        self._fitted = True

        logger.info("Preprocessor.fit_transform: %d rows, %d raw cols -> %d features",
                    X.shape[0], len(self._numeric_cols) + len(self._categorical_cols),
                    X.shape[1])

        if balance:
            X, y = self._apply_smote(X, y)

        return X, y

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Transform without fitting; SMOTE is NEVER applied here (2b
        explicitly restricts balancing to the training partition only)."""
        if not self._fitted:
            raise RuntimeError("Preprocessor.fit_transform() must be called before transform().")

        num_df = df[self._numeric_cols].fillna(0.0)
        cat_df = df[self._categorical_cols].fillna("unknown").astype(str)

        X_num = self.scaler.transform(num_df)
        X_cat = self.encoder.transform(cat_df)
        X = np.hstack([X_num, X_cat])

        y = self._labels_to_indices(df["class"]) if "class" in df.columns else None
        return X, y

    def transform_dataframe(self, X: np.ndarray) -> pd.DataFrame:
        """Convenience: wrap a raw ndarray output back into a labelled
        DataFrame using the fitted output feature names (used by
        FeatureExtractor, which expects column names for MI scoring)."""
        if self.output_feature_names_ is None:
            raise RuntimeError("Preprocessor has not been fitted yet.")
        return pd.DataFrame(X, columns=self.output_feature_names_)

    # ------------------------------------------------------------------
    def _apply_smote(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        class_counts = pd.Series(y).value_counts()
        min_class_count = class_counts.min()

        # SMOTE requires k_neighbors < number of samples in the smallest class
        k_neighbors = min(SMOTE_K_NEIGHBORS, max(min_class_count - 1, 1))
        if min_class_count <= 1:
            logger.warning(
                "Smallest class has only %d sample(s); skipping SMOTE for this "
                "run (cannot synthesize neighbors from a singleton class).",
                min_class_count,
            )
            return X, y

        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_res, y_res = smote.fit_resample(X, y)
        logger.info(
            "SMOTE applied: class counts before=%s after=%s",
            class_counts.to_dict(), pd.Series(y_res).value_counts().to_dict(),
        )
        return X_res, y_res
