"""Tabular fraud classifier with CatBoost."""
from catboost import CatBoostClassifier
import numpy as np


class TabularFraudModel:
    """CatBoost wrapper for tabular fraud features."""

    def __init__(self, class_weight: float = 20):
        self.model = CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="PRAUC",
            class_weights=[1, class_weight],
            verbose=100,
            random_seed=42,
        )

    def fit(self, X, y, X_val=None, y_val=None):
        eval_set = (X_val, y_val) if X_val is not None else None
        self.model.fit(X, y, eval_set=eval_set, early_stopping_rounds=50)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def save(self, path):
        self.model.save_model(str(path))

    @classmethod
    def load(cls, path):
        instance = cls()
        instance.model = CatBoostClassifier()
        instance.model.load_model(str(path))
        return instance
