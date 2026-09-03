import os
import sys
import subprocess
import pandas as pd
import pytest

_SUBMISSION = os.path.join(os.path.dirname(os.path.dirname(__file__)), "FINAL_SUBMISSION")
_REG_SCRIPT = os.path.join(_SUBMISSION, "register.py")
_PAIRS = os.path.join(_SUBMISSION, "verification", "sample_pairs", "pairs.csv")

def test_register_batch(tmp_path):
    out_file = str(tmp_path / "preds.csv")
    cmd = [sys.executable, _REG_SCRIPT, "--input", _PAIRS, "--output", out_file]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"register.py failed: {res.stderr}"
    assert os.path.exists(out_file)
    df = pd.read_csv(out_file)
    assert len(df) == 3
