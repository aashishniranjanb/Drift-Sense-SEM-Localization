import pandas as pd
import numpy as np
from scipy.stats import spearmanr

gt = pd.read_csv('data/phase2_dev/pairs.csv')
p_raw = pd.read_csv('data/phase2_dev/v25_predictions.csv')
p_thresh = pd.read_csv('data/phase2_dev/v25_predictions_thresh.csv')

# Merge raw
m_raw = pd.merge(gt, p_raw, on='pair_id')
m_raw['loc_err'] = np.hypot(m_raw['x'] - m_raw['gt_x'], m_raw['y'] - m_raw['gt_y'])

# Correctness in raw (found=1 everywhere):
# gt_found==1 and loc_err <= 5.0 -> 1
# else -> 0 (either loc_err > 5.0 or gt_found == 0)
y_raw = []
for _, r in m_raw.iterrows():
    if r['gt_found'] == 1 and r['loc_err'] <= 5.0:
        y_raw.append(1)
    else:
        y_raw.append(0)
y_raw = np.array(y_raw)

rho_raw, _ = spearmanr(m_raw['score'], y_raw)
print(f"Number of correct pairs in raw: {np.sum(y_raw)} / 180")
print(f"Spearman rho on raw: {rho_raw:.6f} -> {rho_raw*10:.2f} points")

# Merge thresh
m_thresh = pd.merge(gt, p_thresh, on='pair_id')
m_thresh['loc_err'] = np.hypot(m_thresh['x'] - m_thresh['gt_x'], m_thresh['y'] - m_thresh['gt_y'])

# Correctness in thresh:
# if gt_found==1 and found==1: loc_err <= 5 -> 1, else 0
# if gt_found==0 and found==0: 1 (REJECTION_SUCCESS)
# if gt_found==1 and found==0: 0 (PRESENCE_FALSE_NEGATIVE)
# if gt_found==0 and found==1: 0 (ABSENCE_FALSE_POSITIVE)
y_thresh = []
for _, r in m_thresh.iterrows():
    gf = r['gt_found']
    pf = r['found']
    if gf == 1 and pf == 1:
        y_thresh.append(1 if r['loc_err'] <= 5.0 else 0)
    elif gf == 0 and pf == 0:
        y_thresh.append(1)
    else:
        y_thresh.append(0)
y_thresh = np.array(y_thresh)

rho_thresh, _ = spearmanr(m_thresh['score'], y_thresh)
print(f"\nNumber of correct pairs in thresh: {np.sum(y_thresh)} / 180")
print(f"  - True Positives (present & localized <= 5px): {np.sum((m_thresh['gt_found']==1) & (m_thresh['found']==1) & (m_thresh['loc_err']<=5.0))}")
print(f"  - True Negatives (absent & rejected):          {np.sum((m_thresh['gt_found']==0) & (m_thresh['found']==0))}")
print(f"  - Total correct: {np.sum(y_thresh)}")
print(f"Spearman rho on thresh: {rho_thresh:.6f} -> {rho_thresh*10:.2f} points")
