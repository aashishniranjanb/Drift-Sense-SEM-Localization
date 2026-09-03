import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('phase2/V47_RESEARCH/v47_candidate_cache/features.csv')

# Target: Is the V46 candidate a TRUE match?
# A true match means gt_found=1 and it is within 5px of gt_x, gt_y.
def get_target(r):
    if r['gt_found'] == 0: return 0
    if np.isnan(r['gt_x']) or np.isnan(r['gt_y']): return 0
    d = np.hypot(r['v46_cx'] - r['gt_x'], r['v46_cy'] - r['gt_y'])
    if d <= 5.0: return 1
    return 0

df['is_correct'] = df.apply(get_target, axis=1)

features = [
    'ncc', 'grad', 'ctx', 'phase',
    'ncc_pct', 'grad_pct',
    'prom5_ncc', 'prom10_ncc', 'prom20_ncc',
    'z5_ncc', 'z10_ncc',
    'comp10', 'comp20', 'd1', 'd2',
    'dist_center', 'dist_border',
    'sharpness', 'delta_ncc', 'dist_v25_v46'
]

# Impute NaNs safely
X = df[features].fillna(0).values
y = df['is_correct'].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_lr = np.zeros(len(df))
oof_hgb2 = np.zeros(len(df))
oof_hgb3 = np.zeros(len(df))
oof_hand = np.zeros(len(df))

for train_idx, val_idx in skf.split(X, y):
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    # LR
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    
    lr = LogisticRegression(max_iter=1000, C=0.1, class_weight='balanced')
    lr.fit(X_train_s, y_train)
    oof_lr[val_idx] = lr.predict_proba(X_val_s)[:, 1]
    
    # HGB2
    hgb2 = HistGradientBoostingClassifier(max_depth=2, max_iter=100, learning_rate=0.05, random_state=42)
    hgb2.fit(X_train, y_train)
    oof_hgb2[val_idx] = hgb2.predict_proba(X_val)[:, 1]
    
    # HGB3
    hgb3 = HistGradientBoostingClassifier(max_depth=3, max_iter=100, learning_rate=0.05, random_state=42)
    hgb3.fit(X_train, y_train)
    oof_hgb3[val_idx] = hgb3.predict_proba(X_val)[:, 1]

# Hand gate
def hand_gate(r):
    score = 0.0
    if r['prom10_ncc'] > 0.05: score += 0.2
    if r['prom10_ncc'] > 0.10: score += 0.2
    if r['ncc_pct'] > 99.0: score += 0.2
    if r['sharpness'] > 0.1: score += 0.2
    if r['d1'] > 15.0: score += 0.2 # No close competitors
    return score

df['oof_lr'] = oof_lr
df['oof_hgb2'] = oof_hgb2
df['oof_hgb3'] = oof_hgb3
df['oof_hand'] = df.apply(hand_gate, axis=1)

df.to_csv('phase2/V47_RESEARCH/v47_candidate_cache/features_oof.csv', index=False)
print("Saved OOF predictions to features_oof.csv")
