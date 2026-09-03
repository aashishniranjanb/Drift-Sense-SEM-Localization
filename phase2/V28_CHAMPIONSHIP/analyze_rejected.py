import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import subprocess
import os

gt = pd.read_csv('data/phase2_dev/pairs.csv')
p_thresh = pd.read_csv('data/phase2_dev/v25_predictions_thresh.csv')
feats = pd.read_csv('phase2/V27_FINAL/v25_features.csv')

# Load the unthresholded predictions to know the raw presence score
p_raw = pd.read_csv('data/phase2_dev/v25_predictions.csv')

# Base merged
m = pd.merge(gt, p_thresh, on='pair_id', suffixes=('_gt', '_pred'))
m['raw_score'] = p_raw['score'] # unthresholded presence score
m['margin'] = feats['margin']
m['top1_corr'] = feats['top1_corr']
m['top1_ctx'] = feats['top1_ctx']
m['top1_neigh'] = feats['top1_neigh']
m['top1_grad'] = feats['top1_grad']

m['loc_err'] = np.hypot(m['x'] - m['gt_x'], m['y'] - m['gt_y'])

# Let's inspect the correlation breakdown:
# If found == 1:
#   True Positives (77 pairs): raw_score is high (0.86 to 0.96, mean 0.93).
#   Periodic Replicas (1 pair): raw_score is 0.87.
#   False Positives (2 pairs, Set C): raw_score is 0.90, 0.92.
# If found == 0:
#   True Negatives (38 pairs, Set C): raw_score is 0.56 to 0.84, mean 0.66.
#   False Negatives (62 pairs, Set A/B): raw_score is 0.58 to 0.79, mean 0.65.

# Notice that for found == 0:
# True Negatives are CORRECT (y=1 in benchmark!).
# False Negatives are INCORRECT (y=0 in benchmark!).
# Can we distinguish True Negatives (absent) from False Negatives (present) using features?
tn = m[(m['found'] == 0) & (m['gt_found'] == 0)]
fn = m[(m['found'] == 0) & (m['gt_found'] == 1)]

print(f"Among 100 rejected pairs: {len(tn)} True Negatives (Set C), {len(fn)} False Negatives (Set A/B)")
print("\n--- True Negatives vs False Negatives Feature Comparison ---")
for col in ['raw_score', 'margin', 'top1_corr', 'top1_ctx', 'top1_neigh', 'top1_grad']:
    print(f"{col:12s} | TN mean: {tn[col].mean():.4f}, std: {tn[col].std():.4f} | FN mean: {fn[col].mean():.4f}, std: {fn[col].std():.4f}")
