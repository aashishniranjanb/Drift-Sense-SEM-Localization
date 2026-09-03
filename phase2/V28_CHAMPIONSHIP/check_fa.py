import pandas as pd
import numpy as np

# Load ground truth and predictions
gt = pd.read_csv('data/phase2_dev/pairs.csv')
p_thresh = pd.read_csv('data/phase2_dev/v25_predictions_thresh.csv')
p_raw = pd.read_csv('data/phase2_dev/v25_predictions.csv')
feats = pd.read_csv('phase2/V27_FINAL/v25_features.csv')

# Merge
m = pd.merge(gt, p_thresh, on='pair_id', suffixes=('_gt', '_pred'))
m['raw_score'] = p_raw['score']
m['margin'] = feats['margin']
m['top1_ctx'] = feats['top1_ctx']
m['loc_err'] = np.hypot(m['x'] - m['gt_x'], m['y'] - m['gt_y'])

# Let's inspect the two false accepts in Set C: pair_140 and pair_159
fa = m[m['pair_id'].isin(['pair_140', 'pair_159'])]
print("=== THE 2 FALSE ACCEPTS IN SET C ===")
print(fa[['pair_id', 'raw_score', 'margin', 'top1_ctx']])

# What are their scores vs true accepts?
ta = m[(m['gt_found'] == 1) & (m['found'] == 1)]
print(f"\nTrue accepts min score: {ta['raw_score'].min():.6f}")
print(f"pair_140 raw_score:     {fa.loc[fa['pair_id']=='pair_140', 'raw_score'].values[0]:.6f}")
print(f"pair_159 raw_score:     {fa.loc[fa['pair_id']=='pair_159', 'raw_score'].values[0]:.6f}")
