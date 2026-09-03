import os
import sys
import subprocess
import pytest

_SUBMISSION = os.path.join(os.path.dirname(os.path.dirname(__file__)), "FINAL_SUBMISSION")
_GEN_SCRIPT = os.path.join(_SUBMISSION, "generate_dataset.py")

def test_synthetic_generation(tmp_path):
    out_dir = str(tmp_path / "gen_test")
    cmd = [sys.executable, _GEN_SCRIPT, "--style", "finfet", "--num_pairs", "2", "--output_dir", out_dir, "--seed", "99"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Generator failed: {res.stderr}"
    assert os.path.exists(os.path.join(out_dir, "ground_truth.csv"))
