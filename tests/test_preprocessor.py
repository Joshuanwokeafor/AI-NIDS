"""tests/test_preprocessor.py — unit tests for Stage 2 (Preprocessor)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessor import Preprocessor  # noqa: E402
from config import NUMERIC_FEATURES, CATEGORICAL_FEATURES  # noqa: E402


def _make_df(n_per_class=20):
    rng = np.random.RandomState(42)
    classes = (["normal"] * n_per_class + ["neptune"] * n_per_class
               + ["satan"] * n_per_class + ["guess_passwd"] * (n_per_class // 2)
               + ["rootkit"] * (n_per_class // 4))
    rows = []
    for c in classes:
        row = {f: float(rng.rand() * 100) for f in NUMERIC_FEATURES}
        row["protocol_type"] = rng.choice(["tcp", "udp", "icmp"])
        row["service"] = rng.choice(["http", "ftp", "smtp"])
        row["flag"] = rng.choice(["SF", "S0", "REJ"])
        row["class"] = c
        rows.append(row)
    return pd.DataFrame(rows)


class TestPreprocessor:
    def test_fit_transform_normalizes_to_0_1(self):
        df = _make_df()
        pre = Preprocessor()
        X, y = pre.fit_transform(df, balance=False)
        numeric_slice = X[:, :len(NUMERIC_FEATURES)]
        assert numeric_slice.min() >= 0.0
        assert numeric_slice.max() <= 1.0 + 1e-9

    def test_fit_transform_expands_features_via_one_hot(self):
        df = _make_df()
        pre = Preprocessor()
        X, y = pre.fit_transform(df, balance=False)
        # Should be more columns than raw numeric+categorical due to one-hot
        raw_cols = len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)
        assert X.shape[1] > raw_cols

    def test_smote_balances_classes(self):
        df = _make_df()
        pre = Preprocessor()
        X, y = pre.fit_transform(df, balance=True)
        counts = pd.Series(y).value_counts()
        # After SMOTE all represented classes should have equal counts
        assert counts.nunique() == 1

    def test_smote_only_applied_when_requested(self):
        df = _make_df()
        pre = Preprocessor()
        X, y = pre.fit_transform(df, balance=False)
        counts = pd.Series(y).value_counts()
        assert counts.nunique() > 1  # imbalance preserved

    def test_transform_before_fit_raises(self):
        pre = Preprocessor()
        df = _make_df()
        with pytest.raises(RuntimeError):
            pre.transform(df)

    def test_transform_never_applies_smote(self):
        df = _make_df()
        pre = Preprocessor()
        pre.fit_transform(df, balance=True)
        X, y = pre.transform(df)  # same imbalanced df
        assert X.shape[0] == len(df)  # no synthetic rows added

    def test_unseen_categorical_value_does_not_crash(self):
        train_df = _make_df()
        pre = Preprocessor()
        pre.fit_transform(train_df, balance=False)

        test_df = train_df.copy()
        test_df.loc[0, "service"] = "totally_unseen_service"
        X, y = pre.transform(test_df)  # should not raise (handle_unknown="ignore")
        assert X.shape[0] == len(test_df)

    def test_unknown_attack_label_mapped_not_dropped(self):
        df = _make_df()
        df.loc[0, "class"] = "some_brand_new_zero_day"
        pre = Preprocessor()
        X, y = pre.fit_transform(df, balance=False)
        assert len(y) == len(df)  # record retained, not silently dropped
