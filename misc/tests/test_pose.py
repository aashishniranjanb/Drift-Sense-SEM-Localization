import os
import pandas as pd
import pytest

_SUBMISSION = os.path.join(os.path.dirname(os.path.dirname(__file__)), "FINAL_SUBMISSION")
_SAMPLE_PREDS = os.path.join(_SUBMISSION, "verification", "sample_pairs", "predictions.csv")

def test_pose_ranges_when_found():
    df = pd.read_csv(_SAMPLE_PREDS)
    present = df[df["found"] == 1]
    for _, row in present.iterrows():
        assert -10.0 <= row["theta"] <= 10.0, f"theta unexpected: {row['theta']}"
        assert 5.0 <= row["scale"] <= 15.0, f"scale unexpected: {row['scale']}"
