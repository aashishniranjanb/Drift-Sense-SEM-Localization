import pandas as pd
import numpy as np
import sys
import os

sys.path.append('phase2/V22_CHAMPIONSHIP')
from scorer import compute_competition_score

gt = pd.read_csv('data/phase2_dev/pairs.csv')
pred = pd.read_csv('data/phase2_dev/v25_predictions_thresh.csv')

# Merge
merged = pd.merge(gt, pred, on='pair_id', suffixes=('', '_pred'))
if 'found' in merged.columns and 'pred_found' not in merged.columns:
    merged['pred_found'] = merged['found']

res = compute_competition_score(merged, runtime_median=3.2)
print("=== V25 EVALUATED UNDER V22 SCORER ===")
for k, v in res.items():
    if isinstance(v, float):
        print(f"{k}: {v:.4f}")
    else:
        print(f"{k}: {v}")
