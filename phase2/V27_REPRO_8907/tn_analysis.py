import pandas as pd
import numpy as np

# Let's inspect the 38 true negative pairs (Set C pairs correctly rejected in thresh)
gt = pd.read_csv('data/phase2_dev/pairs.csv')
p_thresh = pd.read_csv('data/phase2_dev/v25_predictions_thresh.csv')
m = pd.merge(gt, p_thresh, on='pair_id')

tn = m[(m['gt_found'] == 0) & (m['found'] == 0)]
tp = m[(m['gt_found'] == 1) & (m['found'] == 1) & (m['x'] > 0)] # present and accepted
fn = m[(m['gt_found'] == 1) & (m['found'] == 0)] # present and rejected

print(f"True Positives (Present, Accepted): {len(tp)}")
print(f"  Score mean: {tp['score'].mean():.4f}, min: {tp['score'].min():.4f}, max: {tp['score'].max():.4f}")

print(f"\nTrue Negatives (Absent, Rejected): {len(tn)}")
print(f"  Score mean: {tn['score'].mean():.4f}, min: {tn['score'].min():.4f}, max: {tn['score'].max():.4f}")

print(f"\nFalse Negatives (Present, Rejected): {len(fn)}")
print(f"  Score mean: {fn['score'].mean():.4f}, min: {fn['score'].min():.4f}, max: {fn['score'].max():.4f}")
