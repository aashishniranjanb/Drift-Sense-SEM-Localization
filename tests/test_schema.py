import os
import pandas as pd
import pytest

_SUBMISSION = os.path.join(os.path.dirname(os.path.dirname(__file__)), "FINAL_SUBMISSION")
_SAMPLE_PREDS = os.path.join(_SUBMISSION, "verification", "sample_pairs", "predictions.csv")

def test_predictions_schema():
    assert os.path.exists(_SAMPLE_PREDS), f"Sample predictions not found: {_SAMPLE_PREDS}"
    df = pd.read_csv(_SAMPLE_PREDS)
    expected = ["pair_id", "x", "y", "theta", "scale", "found", "score"]
    assert list(df.columns) == expected, f"Columns mismatch: {list(df.columns)}"
    assert len(df) > 0, "Empty predictions file"

def test_predictions_types():
    df = pd.read_csv(_SAMPLE_PREDS)
    assert pd.api.types.is_numeric_dtype(df["x"])
    assert pd.api.types.is_numeric_dtype(df["y"])
    assert pd.api.types.is_numeric_dtype(df["theta"])
    assert pd.api.types.is_numeric_dtype(df["scale"])
    assert set(df["found"].unique()).issubset({0, 1})
    assert (df["score"] >= 0.0).all() and (df["score"] <= 1.0).all()
