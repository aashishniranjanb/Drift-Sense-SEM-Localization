#!/usr/bin/env python3
import sys
import os
import time
import subprocess
import pandas as pd

def verify_runtime(input_csv):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reg_py = os.path.join(root, "FINAL_SUBMISSION", "register.py")
    out = os.path.join(root, "JUDGE_TEST", "runtime_tmp.csv")
    
    t0 = time.time()
    res = subprocess.run([sys.executable, reg_py, "--input", input_csv, "--output", out], check=True, stdout=subprocess.PIPE)
    dt = time.time() - t0
    
    df = pd.read_csv(input_csv)
    n_pairs = len(df)
    per_pair = dt / max(1, n_pairs)
    
    if os.path.exists(out): os.remove(out)
    
    if per_pair <= 5.0:
        print(f"[PASS] Runtime budget: {per_pair:0.3f}s / pair (Limit: <= 5.0s)")
        return True
    else:
        print(f"[FAIL] Runtime exceeded: {per_pair:0.3f}s / pair > 5.0s")
        return False

if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else "JUDGE_TEST/sample_pairs/pairs.csv"
    ok = verify_runtime(inp)
    sys.exit(0 if ok else 1)
