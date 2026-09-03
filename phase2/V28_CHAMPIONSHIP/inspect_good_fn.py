import pandas as pd
import numpy as np

# Load ground truth and raw unthresholded predictions
gt = pd.read_csv('data/phase2_dev/pairs.csv')
p_raw = pd.read_csv('data/phase2_dev/v25_predictions.csv')
feats = pd.read_csv('phase2/V27_FINAL/v25_features.csv')

m = pd.merge(gt, p_raw, on='pair_id', suffixes=('_gt', '_pred'))
m['loc_err'] = np.hypot(m['x'] - m['gt_x'], m['y'] - m['gt_y'])

# Let's inspect the 62 false rejects:
# They are pairs where gt_found == 1, but v25_score <= 0.843.
v25_thresh = pd.read_csv('data/phase2_dev/v25_predictions_thresh.csv')
m['v25_found'] = v25_thresh['found']

fn_pairs = m[(m['gt_found'] == 1) & (m['v25_found'] == 0)].copy()
print(f"Total False Rejects in V25: {len(fn_pairs)}")

# How many of these 62 False Rejects have GOOD localization (loc_err <= 5.0 px)?
good_fn = fn_pairs[fn_pairs['loc_err'] <= 5.0]
bad_fn = fn_pairs[fn_pairs['loc_err'] > 5.0]

print(f"False Rejects with loc_err <= 5.0 px: {len(good_fn)} (Rescue Candidates!)")
print(f"False Rejects with loc_err > 5.0 px:  {len(bad_fn)} (Correctly rejected replicas!)")

print("\n--- The Good False Rejects (Can be safely rescued!) ---")
print(good_fn[['pair_id', 'set_type', 'score', 'loc_err']].to_string())
