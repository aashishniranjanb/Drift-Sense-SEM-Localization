#!/usr/bin/env python3
import sys
import os
import pandas as pd
import numpy as np

def verify_schema(csv_path):
    if not os.path.exists(csv_path):
        print(f"[FAIL] Missing predictions file: {csv_path}")
        return False
    df = pd.read_csv(csv_path)
    expected = ["pair_id", "x", "y", "theta", "scale", "found", "score"]
    if list(df.columns) != expected:
        print(f"[FAIL] Expected columns {expected}, got {list(df.columns)}")
        return False
    # Check nulls
    if df.isnull().values.any():
        print("[FAIL] Found NaN values in predictions")
        return False
    # Check found=0 invariant
    for _, r in df[df["found"] == 0].iterrows():
        if r["x"] != 0.0 or r["y"] != 0.0 or r["theta"] != 0.0 or r["scale"] != 0.0:
            print(f"[FAIL] Invariant broken on {r['pair_id']}: found=0 but pose != 0")
            return False
    print("[PASS] 7-column schema & invariants strictly verified")
    return True

if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "JUDGE_TEST/expected/predictions.csv"
    ok = verify_schema(csv_file)
    sys.exit(0 if ok else 1)
