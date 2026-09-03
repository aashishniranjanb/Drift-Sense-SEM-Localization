import pandas as pd
import numpy as np

unthresh = pd.read_csv('phase2/V27_FINAL/v25_unthresholded.csv')
gt = pd.read_csv('data/phase2_dev/pairs.csv')
v25 = pd.read_csv('phase2/V27_FINAL/V25_BASELINE.csv')

m = pd.merge(gt, unthresh, on='pair_id', suffixes=('_gt', '_pred'))
m['v25_found'] = v25['found']
m['v25_score'] = v25['score']
m['new_found'] = (m['score'] > 0.865).astype(int)
m['loc_err'] = np.hypot(m['x'] - m['gt_x'], m['y'] - m['gt_y'])

changes = m[m['v25_found'] != m['new_found']].copy()
print(f"Number of changed pairs at t=0.865: {len(changes)}")
print(changes[['pair_id', 'set_type', 'gt_found', 'v25_found', 'new_found', 'score', 'loc_err']])
