import os
import pandas as pd
import pytest

_SUBMISSION = os.path.join(os.path.dirname(os.path.dirname(__file__)), "FINAL_SUBMISSION")
_SAMPLE_PREDS = os.path.join(_SUBMISSION, "verification", "sample_pairs", "predictions.csv")

def test_zero_coordinates_when_not_found():
    df = pd.read_csv(_SAMPLE_PREDS)
    absent = df[df["found"] == 0]
    for _, row in absent.iterrows():
        assert row["x"] == 0.0, f"x != 0 on found=0: {row['x']}"
        assert row["y"] == 0.0, f"y != 0 on found=0: {row['y']}"
        assert row["theta"] == 0.0, f"theta != 0 on found=0: {row['theta']}"
        assert row["scale"] == 0.0, f"scale != 0 on found=0: {row['scale']}"

def test_coordinate_bounds_when_found():
    df = pd.read_csv(_SAMPLE_PREDS)
    present = df[df["found"] == 1]
    for _, row in present.iterrows():
        assert 0.0 <= row["x"] <= 1000.0, f"x out of bounds: {row['x']}"
        assert 0.0 <= row["y"] <= 1000.0, f"y out of bounds: {row['y']}"
