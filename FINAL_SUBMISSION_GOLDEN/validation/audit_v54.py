import pandas as pd
import numpy as np
import os
import sys

def main():
    baseline_csv = "FINAL_SUBMISSION/predictions.csv"
    v54_csv = "FINAL_SUBMISSION/validation/v54_predictions.csv"
    
    if not os.path.exists(v54_csv):
        print(f"ERROR: {v54_csv} not found. Please run register.py first.")
        sys.exit(1)
        
    df_base = pd.read_csv(baseline_csv)
    df_v54 = pd.read_csv(v54_csv)
    
    print("========================================")
    print("         V54 ACTUAL SCORE AUDIT         ")
    print("========================================")
    
    # STEP 2: Verify properties
    if len(df_v54) != 180:
        print(f"WARNING: V54 has {len(df_v54)} rows, expected 180.")
        
    merged = pd.merge(df_base, df_v54, on="pair_id", suffixes=("_base", "_v54"))
    
    # STEP 5: Localization Regression Audit
    changed_xy = 0
    max_dx = 0.0
    max_dy = 0.0
    
    for _, row in merged.iterrows():
        dx = abs(row['x_base'] - row['x_v54'])
        dy = abs(row['y_base'] - row['y_v54'])
        if dx > 1e-4 or dy > 1e-4:
            changed_xy += 1
            max_dx = max(max_dx, dx)
            max_dy = max(max_dy, dy)
            
    print(f"\n[Localization Regression Audit]")
    print(f"Pairs with changed X/Y : {changed_xy}")
    if changed_xy > 0:
        print(f"Max absolute X change  : {max_dx:.4f}")
        print(f"Max absolute Y change  : {max_dy:.4f}")
        print("WARNING: Localization coordinate boundary was violated!")
    else:
        print("PASS: x/y coordinates are perfectly frozen.")
        
    # STEP 6: Pose Audit
    changed_scale = 0
    max_dscale = 0.0
    for _, row in merged.iterrows():
        if row['found_base'] == 1 and row['found_v54'] == 1:
            dscale = abs(row['scale_base'] - row['scale_v54'])
            if dscale > 1e-6:
                changed_scale += 1
                max_dscale = max(max_dscale, dscale)
    
    print(f"\n[Pose Audit]")
    print(f"Pairs with refined scale: {changed_scale}")
    print(f"Max scale adjustment    : {max_dscale:.6f}")
    
    # STEP 7: Rejection Audit
    print(f"\n[Rejection Decision Changes]")
    b0_v1 = merged[(merged['found_base'] == 0) & (merged['found_v54'] == 1)]
    b1_v0 = merged[(merged['found_base'] == 1) & (merged['found_v54'] == 0)]
    
    print(f"Baseline rejected -> V54 accepted : {len(b0_v1)}")
    print(f"Baseline accepted -> V54 rejected : {len(b1_v0)}")
    
    # STEP 8: Calibration Audit
    print(f"\n[Calibration Audit]")
    diff_score = np.mean(np.abs(merged['score_base'] - merged['score_v54']))
    print(f"Mean absolute confidence shift: {diff_score:.4f}")
    
    print("\nNOTE: To compute the exact 90-100 official score, run the official benchmark_phase2.py")
    print("using the validation/v54_predictions.csv and data/phase2_dev/pairs.csv ground truth.")
    print("========================================")

if __name__ == "__main__":
    main()
