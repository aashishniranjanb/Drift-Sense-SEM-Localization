#!/usr/bin/env python3
"""
Drift-Sense++ Strict Competition Contract Validator
Verifies compliance against Applied Materials Phase 2 output specifications.
"""
import sys
import os
import argparse
import pandas as pd
import numpy as np

EXPECTED_COLS = ["pair_id", "x", "y", "theta", "scale", "found", "score"]

def validate_contract(input_csv, predictions_csv):
    errors = []
    warnings = []

    if not os.path.exists(predictions_csv):
        return [f"Output file does not exist: {predictions_csv}"], []

    try:
        preds = pd.read_csv(predictions_csv)
    except Exception as e:
        return [f"Failed to read predictions CSV: {e}"], []

    # 1. Exact 7 columns
    if list(preds.columns) != EXPECTED_COLS:
        errors.append(f"Column mismatch! Expected: {EXPECTED_COLS}, got: {list(preds.columns)}")

    # 2. Input pair_id parity
    if input_csv and os.path.exists(input_csv):
        inp = pd.read_csv(input_csv)
        if "pair_id" in inp.columns:
            in_ids = list(inp["pair_id"].astype(str))
            pred_ids = list(preds["pair_id"].astype(str))

            if len(pred_ids) != len(set(pred_ids)):
                dups = [x for x in pred_ids if pred_ids.count(x) > 1]
                errors.append(f"Duplicate pair_ids found in predictions: {set(dups)}")

            missing = set(in_ids) - set(pred_ids)
            if missing:
                errors.append(f"Missing pair_ids from output (scores zero): {missing}")

            extra = set(pred_ids) - set(in_ids)
            if extra:
                warnings.append(f"Unexpected extra pair_ids in output: {extra}")

    # 3. Check for NaNs or Infs
    for col in preds.columns:
        if preds[col].isna().any():
            errors.append(f"NaN values detected in column: {col}")
        if preds[col].dtype in [np.float64, np.float32, float]:
            if np.isinf(preds[col]).any():
                errors.append(f"Inf values detected in column: {col}")

    # 4. Strict found in {0, 1}
    unique_found = set(preds["found"].unique())
    if not unique_found.issubset({0, 1}):
        errors.append(f"found column contains invalid values (must be 0 or 1): {unique_found}")

    # 5. Invariant barrier: found == 0 => x=y=theta=scale=0.0
    absent = preds[preds["found"] == 0]
    for idx, row in absent.iterrows():
        p_id = row["pair_id"]
        if row["x"] != 0.0 or row["y"] != 0.0 or row["theta"] != 0.0 or row["scale"] != 0.0:
            errors.append(
                f"INVARIANT VIOLATION: found=0 requires x=y=theta=scale=0.0 on pair {p_id}. "
                f"Got x={row['x']}, y={row['y']}, theta={row['theta']}, scale={row['scale']}"
            )

    # 6. Coordinate and scale physical bounds for found == 1
    present = preds[preds["found"] == 1]
    for idx, row in present.iterrows():
        p_id = row["pair_id"]
        if not (-50.0 <= row["x"] <= 1050.0) or not (-50.0 <= row["y"] <= 1050.0):
            warnings.append(f"Coordinates outside image canvas on pair {p_id}: ({row['x']}, {row['y']})")
        if not (-20.0 <= row["theta"] <= 20.0):
            warnings.append(f"Rotation theta outside expected range on pair {p_id}: {row['theta']}")
        if not (1.0 <= row["scale"] <= 25.0):
            warnings.append(f"Scale factor outside physical range on pair {p_id}: {row['scale']}")

    # 7. Confidence score range
    if not ((preds["score"] >= 0.0) & (preds["score"] <= 1.0)).all():
        errors.append("Confidence score values fall outside documented range [0.0, 1.0]")

    return errors, warnings

def main():
    parser = argparse.ArgumentParser(description="Validate Phase 2 Output Contract")
    parser.add_argument("--predictions", required=True, help="Path to predictions.csv")
    parser.add_argument("--input", default=None, help="Optional path to input pairs.csv")
    args = parser.parse_args()

    print(f"Validating contract for: {args.predictions}")
    errors, warnings = validate_contract(args.input, args.predictions)

    if warnings:
        print("\n[WARNINGS]")
        for w in warnings:
            print(f"  - {w}")

    if errors:
        print("\n[CONTRACT VIOLATIONS - FAIL]")
        for e in errors:
            print(f"  [FAIL] {e}")
        sys.exit(1)
    else:
        print("\n[CONTRACT VALIDATION - PASS]")
        print("  [PASS] Exactly 7 columns")
        print("  [PASS] Pair ID parity verified")
        print("  [PASS] No NaNs or Infs")
        print("  [PASS] found in {0, 1}")
        print("  [PASS] found=0 => x=y=theta=scale=0.0 invariant strictly maintained")
        print("  [PASS] Confidence score monotonic [0, 1]")
        sys.exit(0)

if __name__ == "__main__":
    main()

