import pandas as pd
import numpy as np

# Load ground truth and baseline
gt = pd.read_csv('data/phase2_dev/pairs.csv')
v25 = pd.read_csv('phase2/V27_FINAL/V25_BASELINE.csv')
feats = pd.read_csv('phase2/V27_FINAL/v25_features.csv')

# Merge to get all ground truth and predictions
df = pd.merge(gt, v25, on='pair_id', suffixes=('_gt', '_pred'))
df = pd.merge(df, feats[['pair_id', 'top1_score', 'margin', 'top1_corr', 'top1_ctx', 'top1_neigh', 'top1_grad', 'mode_strong']], on='pair_id')

# Note that for rows where v25 found=0, x/y/theta/scale are 0.
# We also have the extracted best candidate coordinates from when v25 ran:
# Wait, let's see how many present pairs were found=1 in v25:
print("V25 found distribution:")
print(df['found'].value_counts())
print("\nV25 found vs gt_found:")
print(pd.crosstab(df['gt_found'], df['found']))

# Check if we have unmasked candidate coordinates in feats:
# In feats: pred_x, pred_y were extracted before masking!
feats_full = pd.read_csv('phase2/V27_FINAL/v25_features.csv')
print("\nfeats pred_x non-zero count:", (feats_full['pred_x'] > 0).sum())
