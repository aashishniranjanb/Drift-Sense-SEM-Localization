import os
import sys
import subprocess
import pytest

_SUBMISSION = os.path.join(os.path.dirname(os.path.dirname(__file__)), "FINAL_SUBMISSION")
_INF_SCRIPT = os.path.join(_SUBMISSION, "inference.py")
_REF = os.path.join(_SUBMISSION, "verification", "sample_pairs", "images", "ref_val_dram_000.png")
_SEARCH = os.path.join(_SUBMISSION, "verification", "sample_pairs", "images", "search_val_dram_000.png")

def test_inference_cli():
    if not os.path.exists(_REF) or not os.path.exists(_SEARCH):
        pytest.skip("Sample test images not found")
    cmd = [sys.executable, _INF_SCRIPT, "--reference", _REF, "--search", _SEARCH]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"inference.py failed: {res.stderr}"
    lines = [l.strip() for l in res.stdout.strip().splitlines() if l.strip()]
    assert any(l.startswith("x=") for l in lines)
    assert any(l.startswith("y=") for l in lines)
