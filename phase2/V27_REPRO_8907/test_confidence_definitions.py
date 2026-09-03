import pandas as pd
import numpy as np
from scipy.stats import spearmanr

gt = pd.read_csv('data/phase2_dev/pairs.csv')
p_thresh = pd.read_csv('data/phase2_dev/v25_predictions_thresh.csv')
m = pd.merge(gt, p_thresh, on='pair_id')
m['loc_err'] = np.hypot(m['x'] - m['gt_x'], m['y'] - m['gt_y'])

# Standard y_thresh
y_thresh = []
for _, r in m.iterrows():
    if r['gt_found'] == 1 and r['found'] == 1:
        y_thresh.append(1 if r['loc_err'] <= 5.0 else 0)
    elif r['gt_found'] == 0 and r['found'] == 0:
        y_thresh.append(1)
    else:
        y_thresh.append(0)
y_thresh = np.array(y_thresh)

# What if confidence score for rejected pairs represents confidence in the REJECTION decision?
# i.e. if found == 0, confidence in rejection is (1.0 - presence_score) or high!
scores_cal = m['score'].copy()
# If we test assigning high confidence to strong rejections:
scores_cal_rej = scores_cal.copy()
# When found == 0, confidence of decision being correct:
scores_cal_rej[m['found'] == 0] = 1.0 - scores_cal[m['found'] == 0]
rho_inv, _ = spearmanr(scores_cal_rej, y_thresh)
print(f"Spearman if score = P(decision is correct): {rho_inv:.4f}")

# What if score for found==0 is fixed or scaled?
for rej_val in [0.0, 0.5, 0.8, 0.9, 0.95, 1.0]:
    s_test = scores_cal.copy()
    s_test[m['found'] == 0] = rej_val
    rho_t, _ = spearmanr(s_test, y_thresh)
    print(f"Rejection score = {rej_val}: rho = {rho_t:.4f} ({rho_t*10:.2f} pts)")
