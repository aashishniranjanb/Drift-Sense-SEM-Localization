import pandas as pd
import numpy as np

# Let's inspect the 80 pairs accepted by V25
v25_thresh = pd.read_csv('phase2/V27_FINAL/V25_BASELINE.csv')
gt = pd.read_csv('data/phase2_dev/pairs.csv')
m = pd.merge(gt, v25_thresh, on='pair_id')

accepted = m[m['found'] == 1]
print(f"Accepted count: {len(accepted)}")
print("Accepted breakdown by ground truth:")
print(accepted['gt_found'].value_counts())

# Among present accepted:
pres_acc = accepted[accepted['gt_found'] == 1].copy()
pres_acc['loc_err'] = np.hypot(pres_acc['x'] - pres_acc['gt_x'], pres_acc['y'] - pres_acc['gt_y'])
print(f"Accepted present loc_err <= 1.0: {(pres_acc['loc_err'] <= 1.0).sum()}/{len(pres_acc)}")
print(f"Accepted present loc_err <= 5.0: {(pres_acc['loc_err'] <= 5.0).sum()}/{len(pres_acc)}")
print(f"Accepted present loc_err > 5.0: {(pres_acc['loc_err'] > 5.0).sum()}/{len(pres_acc)}")

# What is the single periodic replica in V25 accepted?
bad_acc = pres_acc[pres_acc['loc_err'] > 5.0]
print("\nThe 1 accepted pair with > 5px error in V25:")
print(bad_acc[['pair_id', 'set_type', 'loc_err', 'score']])

# What are the 2 false accepts in Set C?
fa = accepted[accepted['gt_found'] == 0]
print("\nThe 2 false accepts (Set C):")
print(fa[['pair_id', 'set_type', 'score']])
