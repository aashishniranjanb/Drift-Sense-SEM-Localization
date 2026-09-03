import pandas as pd
import numpy as np

# Load unthresholded and ground truth
unthresh = pd.read_csv('phase2/V27_FINAL/v25_unthresholded.csv')
gt = pd.read_csv('data/phase2_dev/pairs.csv')
m = pd.merge(gt, unthresh, on='pair_id', suffixes=('_gt', '_pred'))
m['loc_err'] = np.hypot(m['x'] - m['gt_x'], m['y'] - m['gt_y'])

# Let's inspect the distribution of scores for:
# 1. Correct localizations (present, loc_err <= 5)
# 2. Wrong localizations (present, loc_err > 5)
# 3. Absent pairs (gt_found == 0)

correct_loc = m[(m['gt_found'] == 1) & (m['loc_err'] <= 5.0)]
wrong_loc = m[(m['gt_found'] == 1) & (m['loc_err'] > 5.0)]
absent = m[m['gt_found'] == 0]

print(f"Correct localizations (<= 5px): {len(correct_loc)}")
print("Score quantiles:")
print(correct_loc['score'].quantile([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]))

print(f"\nWrong localizations (> 5px): {len(wrong_loc)}")
print("Score quantiles:")
print(wrong_loc['score'].quantile([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]))

print(f"\nAbsent pairs: {len(absent)}")
print("Score quantiles:")
print(absent['score'].quantile([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]))
