import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

# Load ground truth, baseline, and features
gt = pd.read_csv('data/phase2_dev/pairs.csv')
v25 = pd.read_csv('phase2/V27_FINAL/V25_BASELINE.csv')
unthresh = pd.read_csv('phase2/V27_FINAL/v25_unthresholded.csv')
feats = pd.read_csv('phase2/V27_FINAL/v25_features.csv')

m = pd.merge(gt, unthresh, on='pair_id', suffixes=('_gt', '_pred'))
m = pd.merge(m, feats[['pair_id', 'top1_score', 'margin', 'top1_corr', 'top1_ctx', 'top1_neigh', 'top1_grad', 'mode_strong']], on='pair_id')
m['loc_err'] = np.hypot(m['x'] - m['gt_x'], m['y'] - m['gt_y'])
m['v25_found'] = v25['found']
m['v25_score'] = v25['score']

# Define ground-truth correctness label y under V25 decisions
# 1 if SUBPIXEL_SUCCESS (<=1) or IN_BOUNDS_SUCCESS (<=5) or REJECTION_SUCCESS (gt=0, pred=0)
correctness = []
for idx, row in m.iterrows():
    gf = row['gt_found']
    pf = row['v25_found']
    if gf == 1 and pf == 1:
        is_corr = 1 if row['loc_err'] <= 5.0 else 0
    elif gf == 0 and pf == 0:
        is_corr = 1
    else:
        is_corr = 0
    correctness.append(is_corr)
m['y_true'] = correctness

# S0: V25 score
s0 = m['v25_score'].values
rho0, _ = spearmanr(s0, m['y_true'])

# S1: V25 score + margin
# Let's test a simple linear blend
s1 = m['v25_score'].values + 0.1 * m['margin'].values
rho1, _ = spearmanr(s1, m['y_true'])

# S2: V25 score + margin + context
s2 = m['v25_score'].values + 0.1 * m['margin'].values + 0.05 * m['top1_ctx'].values
rho2, _ = spearmanr(s2, m['y_true'])

# S3: Logistic Regression using V25 features
feature_cols = ['top1_score', 'margin', 'top1_corr', 'top1_ctx', 'top1_neigh', 'top1_grad', 'mode_strong']
lr = LogisticRegression(max_iter=1000)
lr.fit(m[feature_cols], m['y_true'])
s3 = lr.predict_proba(m[feature_cols])[:, 1]
rho3, _ = spearmanr(s3, m['y_true'])

# S4: 5-Fold OOF Logistic Regression
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
s4 = np.zeros(len(m))
for tr_idx, val_idx in skf.split(m, m['y_true']):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(m.iloc[tr_idx][feature_cols], m.iloc[tr_idx]['y_true'])
    s4[val_idx] = clf.predict_proba(m.iloc[val_idx][feature_cols])[:, 1]
rho4, _ = spearmanr(s4, m['y_true'])

calib_df = pd.DataFrame([
    {'Score_Name': 'S0: V25 Score', 'Spearman_rho': rho0, 'Calibration_Points': rho0 * 10.0, 'Notes': 'Untouched frozen V25'},
    {'Score_Name': 'S1: V25 + Margin', 'Spearman_rho': rho1, 'Calibration_Points': rho1 * 10.0, 'Notes': 'Score + 0.1*margin'},
    {'Score_Name': 'S2: V25 + Margin + Context', 'Spearman_rho': rho2, 'Calibration_Points': rho2 * 10.0, 'Notes': 'Score + 0.1*margin + 0.05*context'},
    {'Score_Name': 'S3: Full LR (In-Sample)', 'Spearman_rho': rho3, 'Calibration_Points': rho3 * 10.0, 'Notes': 'Logistic regression fitted on all 180'},
    {'Score_Name': 'S4: OOF LR (5-Fold CV)', 'Spearman_rho': rho4, 'Calibration_Points': rho4 * 10.0, 'Notes': 'Out-of-fold generalizable estimate'}
])

calib_df.to_csv('phase2/V27_FINAL/calibration_comparison.csv', index=False)
print("=== CALIBRATION COMPARISON ===")
print(calib_df)
