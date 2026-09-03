import pandas as pd
import numpy as np

# Let's inspect the correlation when we assign lower scores to rejected predictions
gt = pd.read_csv('data/phase2_dev/pairs.csv')
p_thresh = pd.read_csv('data/phase2_dev/v25_predictions_thresh.csv')
m = pd.merge(gt, p_thresh, on='pair_id')
m['loc_err'] = np.hypot(m['x'] - m['gt_x'], m['y'] - m['gt_y'])

from scipy.stats import spearmanr

# 1. Official benchmark calculation on v25_predictions_thresh:
y_thresh = []
for _, r in m.iterrows():
    if r['gt_found'] == 1 and r['found'] == 1:
        y_thresh.append(1 if r['loc_err'] <= 5.0 else 0)
    elif r['gt_found'] == 0 and r['found'] == 0:
        y_thresh.append(1)
    else:
        y_thresh.append(0)
y_thresh = np.array(y_thresh)

rho_base, _ = spearmanr(m['score'], y_thresh)
print(f"Base thresh Spearman: {rho_base:.4f} ({rho_base*10:.2f} pts)")

# 2. What if score reflects the rejection decision? (e.g. if found==0, score is 0.0 or 1.0 - pres_score)
# In standard detection, the score column is the confidence that FOUND=1!
# If the prediction says FOUND=0, its confidence of being PRESENT is low.
# But what if confidence is P(FOUND == 1)? Then absent pairs should have LOW score.
# And they DO have low score! (mean 0.66).
# But benchmark defines correctness = 1 for REJECTION_SUCCESS!
# So benchmark assigns y=1 to pairs with LOW scores!
print("\nCorrelation breakdown:")
print(f"Mean score when y=1 (TP + TN): {m.loc[y_thresh==1, 'score'].mean():.4f}")
print(f"  - TP subset of y=1: {m.loc[(m['gt_found']==1) & (m['found']==1) & (m['loc_err']<=5.0), 'score'].mean():.4f}")
print(f"  - TN subset of y=1: {m.loc[(m['gt_found']==0) & (m['found']==0), 'score'].mean():.4f}")
print(f"Mean score when y=0 (FP + FN): {m.loc[y_thresh==0, 'score'].mean():.4f}")
