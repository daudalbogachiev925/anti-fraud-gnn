"""Data loading. IEEE-CIS format, falls back to synthetic."""
import pandas as pd
import numpy as np
from src.config import DATA_DIR, SEED, FRAUD_RATE


def load_transactions() -> pd.DataFrame:
    """
    Load IEEE-CIS or synthetic.

    Real data: https://www.kaggle.com/competitions/ieee-fraud-detection
    Place train_transaction.csv in data/raw/
    """
    real = DATA_DIR / "raw" / "train_transaction.csv"
    if real.exists():
        return pd.read_csv(real)

    print("Real data not found, generating synthetic.")
    return _generate_synthetic()


def _generate_synthetic(n: int = 50000) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    n_fraud = int(n * FRAUD_RATE)
    n_legit = n - n_fraud

    legit = pd.DataFrame({
        "TransactionID": range(n_legit),
        "user_id": rng.integers(0, 10000, n_legit),
        "card_id": rng.integers(0, 5000, n_legit),
        "merchant_id": rng.integers(0, 500, n_legit),
        "device_id": rng.integers(0, 8000, n_legit),
        "amount": rng.lognormal(4, 1, n_legit).clip(1, 5000),
        "hour": rng.integers(0, 24, n_legit),
        "is_fraud": 0,
    })

    fraud = pd.DataFrame({
        "TransactionID": range(n_legit, n),
        "user_id": rng.integers(0, 10000, n_fraud),
        "card_id": rng.integers(0, 5000, n_fraud),
        "merchant_id": rng.integers(0, 500, n_fraud),
        "device_id": rng.integers(0, 8000, n_fraud),
        "amount": rng.lognormal(5, 1.5, n_fraud).clip(1, 20000),
        "hour": rng.integers(0, 24, n_fraud),
        "is_fraud": 1,
    })

    return pd.concat([legit, fraud]).sample(frac=1, random_state=SEED).reset_index(drop=True)
