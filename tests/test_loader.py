"""Tests for data loader."""
from src.data.loader import load_transactions


def test_loader_returns_dataframe():
    df = load_transactions()
    assert len(df) > 0
    assert "is_fraud" in df.columns
    assert set(df["is_fraud"].unique()).issubset({0, 1})


def test_loader_has_required_columns():
    df = load_transactions()
    for col in ["user_id", "card_id", "merchant_id", "device_id", "amount"]:
        assert col in df.columns
