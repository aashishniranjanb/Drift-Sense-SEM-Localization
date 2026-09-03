import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import spearmanr
import pickle

df = pd.read_csv('phase2/V27_REJECTION/oof_predictions.csv')
v25_preds = pd.read_csv('data/phase2_dev/v25_predictions_thresh.csv')
df = pd.merge(df, v25_preds[['pair_id', 'x', 'y']], on='pair_id')
pairs = pd.read_csv('data/phase2_dev/pairs.csv')
merged = pd.merge(pairs, df, on='pair_id', suffixes=('', '_df'))
merged['loc_err'] = np.hypot(merged['x'] - merged['gt_x'], merged['y'] - merged['gt_y'])

# Let's say we freeze the rejection threshold at 0.68
merged['pred_found'] = (merged['oof_combined'] > 0.68).astype(int)

correctness = []
for _, row in merged.iterrows():
    gt = row['gt_found']
    pr = row['pred_found']
    if gt == 1 and pr == 1:
        correctness.append(1 if row['loc_err'] <= 5.0 else 0)
    elif gt == 0 and pr == 0:
        correctness.append(1)
    else:
        correctness.append(0)
merged['is_correct'] = correctness

features = ['oof_combined', 'top1_score', 'margin', 'top1_corr', 'top1_ctx', 'top1_neigh', 'top1_grad', 'mode_strong']

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_calib = np.zeros(len(merged))

for train_idx, val_idx in skf.split(merged, merged['is_correct']):
    X_train, X_val = merged.iloc[train_idx][features], merged.iloc[val_idx][features]
    y_train = merged.iloc[train_idx]['is_correct']
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    oof_calib[val_idx] = clf.predict_proba(X_val)[:, 1]

merged['calib_score'] = oof_calib

s_raw, _ = spearmanr(merged['oof_combined'], merged['is_correct'])
s_lr, _ = spearmanr(merged['calib_score'], merged['is_correct'])

print(f"OOF Combined Spearman: {s_raw:.4f}")
print(f"LR Multi-Feature Spearman: {s_lr:.4f}")

final_lr = LogisticRegression(max_iter=1000)
final_lr.fit(merged[features], merged['is_correct'])
with open('phase2/V27_CALIBRATION/calib_model.pkl', 'wb') as f:
    pickle.dump({'model': final_lr, 'features': features}, f)
