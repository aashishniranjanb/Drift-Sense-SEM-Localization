import pandas as pd
import numpy as np

# Load unthresholded V25 predictions, ground truth, and features
gt = pd.read_csv('data/phase2_dev/pairs.csv')
unthresh = pd.read_csv('phase2/V27_FINAL/v25_unthresholded.csv')
feats = pd.read_csv('phase2/V27_FINAL/v25_features.csv')

# Merge
df = pd.merge(gt, unthresh, on='pair_id', suffixes=('_gt', '_pred'))
df = pd.merge(df, feats[['pair_id', 'top1_score', 'margin', 'top1_corr', 'top1_ctx', 'top1_neigh', 'top1_grad', 'mode_strong']], on='pair_id')

# Calculate localization error for all pairs (using unthresholded coordinates)
df['loc_err'] = np.hypot(df['x'] - df['gt_x'], df['y'] - df['gt_y'])

print(f"Total pairs: {len(df)}")
print("Summary of unthresholded scores:")
print(df['score'].describe())

# Check how many pairs are present, and what their loc_err is:
present = df[df['gt_found'] == 1]
print(f"\nPresent pairs count: {len(present)}")
print(f"Present pairs with loc_err <= 1.0: {(present['loc_err'] <= 1.0).sum()}")
print(f"Present pairs with loc_err <= 5.0: {(present['loc_err'] <= 5.0).sum()}")
print(f"Present pairs with loc_err > 5.0: {(present['loc_err'] > 5.0).sum()}")

# Check absent pairs
absent = df[df['gt_found'] == 0]
print(f"\nAbsent pairs count: {len(absent)}")
