import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

gt = pd.read_csv('data/phase2_dev/pairs.csv')
p_thresh = pd.read_csv('data/phase2_dev/v25_predictions_thresh.csv')
p_raw = pd.read_csv('data/phase2_dev/v25_predictions.csv')
feats = pd.read_csv('phase2/V27_FINAL/v25_features.csv')

m = pd.merge(gt, p_thresh, on='pair_id', suffixes=('_gt', '_pred'))
m['raw_score'] = p_raw['score']
m['loc_err'] = np.hypot(m['x'] - m['gt_x'], m['y'] - m['gt_y'])
for c in ['margin', 'top1_corr', 'top1_ctx', 'top1_neigh', 'top1_grad', 'mode_strong']:
    m[c] = feats[c]

# Benchmark correctness label y:
y = []
for _, r in m.iterrows():
    gf, pf = r['gt_found'], r['found']
    if gf == 1 and pf == 1:
        y.append(1 if r['loc_err'] <= 5.0 else 0)
    elif gf == 0 and pf == 0:
        y.append(1)
    else:
        y.append(0)
m['y_correct'] = np.array(y)

# Let's inspect: among rejected pairs (found == 0), how well can we predict y_correct?
m_rej = m[m['found'] == 0].copy()
y_rej = m_rej['y_correct'].values # 1 if TN (Set C), 0 if FN (Set A/B)
print(f"Rejected pairs count: {len(m_rej)}, y=1 count: {y_rej.sum()} (38 TN), y=0 count: {(1-y_rej).sum()} (62 FN)")

# 5-fold CV Logistic Regression on rejected pairs
feature_cols = ['raw_score', 'margin', 'top1_corr', 'top1_ctx', 'top1_neigh', 'top1_grad', 'mode_strong']
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_rej_prob = np.zeros(len(m_rej))

for tr, val in skf.split(m_rej, y_rej):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(m_rej.iloc[tr][feature_cols], y_rej[tr])
    oof_rej_prob[val] = clf.predict_proba(m_rej.iloc[val][feature_cols])[:, 1]

from sklearn.metrics import roc_auc_score
print(f"OOF ROC-AUC of predicting correct rejection: {roc_auc_score(y_rej, oof_rej_prob):.4f}")

# What if we train on all 100 rejected pairs?
clf_all = LogisticRegression(max_iter=1000)
clf_all.fit(m_rej[feature_cols], y_rej)
in_sample_prob = clf_all.predict_proba(m_rej[feature_cols])[:, 1]
print(f"In-sample ROC-AUC of predicting correct rejection: {roc_auc_score(y_rej, in_sample_prob):.4f}")
