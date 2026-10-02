"""Ensemble of tabular and graph models via logistic regression."""
import numpy as np
from sklearn.linear_model import LogisticRegression


class FraudEnsemble:
    """Stacking: combine tabular probs, GNN embeddings, and raw features."""

    def __init__(self):
        self.meta = LogisticRegression(max_iter=1000, class_weight="balanced")

    def fit(self, meta_features: np.ndarray, y: np.ndarray):
        self.meta.fit(meta_features, y)
        return self

    def predict_proba(self, meta_features: np.ndarray) -> np.ndarray:
        return self.meta.predict_proba(meta_features)[:, 1]
