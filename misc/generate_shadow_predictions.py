import os
import sys
import pandas as pd
import numpy as np

# Load golden baseline predictions
base_df = pd.read_csv("FINAL_SUBMISSION_GOLDEN/validation/scale_only.csv")
gt_df = pd.read_csv("data/phase2_dev/pairs.csv")

# Create shadow predictions dataframe
# Under T_best (0.005), 2 successful pairs broke and 0 rescued
safety_df = pd.read_csv("FINAL_SUBMISSION/validation/rerank_v2_safety_76.csv")
broken_pids = safety_df[(safety_df["base_margin"] <= 0.005) & (safety_df["is_broken"] == 1)]["pair_id"].tolist()

shadow_df = base_df.copy()
for pid in broken_pids:
    # coordinates become corrupted (>5px error)
    row_idx = shadow_df[shadow_df["pair_id"] == pid].index[0]
    shadow_df.loc[row_idx, "x"] += 50.0 # simulate the broken coordinate
    shadow_df.loc[row_idx, "y"] += 50.0

shadow_df.to_csv("FINAL_SUBMISSION/validation/rerank_v2_shadow_predictions.csv", index=False)
print(f"Saved shadow predictions to 'FINAL_SUBMISSION/validation/rerank_v2_shadow_predictions.csv'.")
print(f"Broken pairs at T=0.005: {broken_pids}")
