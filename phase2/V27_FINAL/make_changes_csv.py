import pandas as pd
import numpy as np

# Load unthresholded and ground truth
unthresh = pd.read_csv('phase2/V27_FINAL/v25_unthresholded.csv')
gt = pd.read_csv('data/phase2_dev/pairs.csv')
v25 = pd.read_csv('phase2/V27_FINAL/V25_BASELINE.csv')

m = pd.merge(gt, unthresh, on='pair_id', suffixes=('_gt', '_pred'))
m['v25_found'] = v25['found']
m['v25_score'] = v25['score']
m['new_found'] = (m['score'] > 0.875).astype(int)
m['loc_err'] = np.hypot(m['x'] - m['gt_x'], m['y'] - m['gt_y'])

# Extract only changed rows
changes = m[m['v25_found'] != m['new_found']].copy()

# Classify change_type
def classify(row):
    if row['v25_found'] == 1 and row['new_found'] == 0:
        if row['gt_found'] == 0:
            return 'CORRECT_REJECTION'
        elif row['loc_err'] > 5.0:
            return 'BAD_LOCALIZATION_REJECTED'
        else:
            return 'FALSE_REJECTION'
    elif row['v25_found'] == 0 and row['new_found'] == 1:
        if row['gt_found'] == 1 and row['loc_err'] <= 5.0:
            return 'CORRECT_ACCEPT'
        else:
            return 'FALSE_ACCEPT'
    return 'UNCHANGED'

changes['change_type'] = changes.apply(classify, axis=1)

out_df = changes[['pair_id', 'v25_found', 'new_found', 'v25_score', 'score', 'gt_found', 'loc_err', 'change_type']]
out_df.columns = ['pair_id', 'v25_found', 'new_found', 'v25_score', 'new_score', 'gt_found', 'loc_error', 'change_type']
out_df.to_csv('phase2/V27_FINAL/decision_changes.csv', index=False)
print("Saved decision_changes.csv:")
print(out_df)
