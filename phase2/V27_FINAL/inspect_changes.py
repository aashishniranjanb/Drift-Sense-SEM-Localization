import pandas as pd
import numpy as np

# Let's inspect what happens at threshold t=0.875
unthresh = pd.read_csv('phase2/V27_FINAL/v25_unthresholded.csv')
gt = pd.read_csv('data/phase2_dev/pairs.csv')
v25 = pd.read_csv('phase2/V27_FINAL/V25_BASELINE.csv')

m = pd.merge(gt, unthresh, on='pair_id', suffixes=('_gt', '_pred'))
m['v25_found'] = v25['found']
m['v25_score'] = v25['score']
m['new_found'] = (m['score'] > 0.875).astype(int)
m['loc_err'] = np.hypot(m['x'] - m['gt_x'], m['y'] - m['gt_y'])

changes = m[m['v25_found'] != m['new_found']].copy()
print(f"Number of changed pairs: {len(changes)}")
print(changes[['pair_id', 'set_type', 'gt_found', 'v25_found', 'new_found', 'score', 'loc_err']])

# What was pair_098?
p98 = m[m['pair_id'] == 'pair_098']
print("\npair_098 details:")
print(p98[['pair_id', 'set_type', 'gt_found', 'v25_found', 'new_found', 'score', 'loc_err']])
