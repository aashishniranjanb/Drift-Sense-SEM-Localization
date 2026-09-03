#!/usr/bin/env python3
import sys
import os
import subprocess
import hashlib
import pandas as pd

def get_hash(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def verify_determinism(input_csv):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reg_py = os.path.join(root, "FINAL_SUBMISSION", "register.py")
    out1 = os.path.join(root, "JUDGE_TEST", "det1.csv")
    out2 = os.path.join(root, "JUDGE_TEST", "det2.csv")
    
    subprocess.run([sys.executable, reg_py, "--input", input_csv, "--output", out1], check=True, stdout=subprocess.PIPE)
    subprocess.run([sys.executable, reg_py, "--input", input_csv, "--output", out2], check=True, stdout=subprocess.PIPE)
    
    h1 = get_hash(out1)
    h2 = get_hash(out2)
    
    if os.path.exists(out1): os.remove(out1)
    if os.path.exists(out2): os.remove(out2)
    
    if h1 == h2:
        print(f"[PASS] Deterministic output confirmed (SHA-256 match: {h1[:12]}...)")
        return True
    else:
        print(f"[FAIL] Non-deterministic! Hash 1: {h1} != Hash 2: {h2}")
        return False

if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else "JUDGE_TEST/sample_pairs/pairs.csv"
    ok = verify_determinism(inp)
    sys.exit(0 if ok else 1)
