import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import subprocess, sys

v39 = pd.read_csv("phase2/V39_POSE/v39_predictions.csv")
pairs = pd.read_csv("data/phase2_dev/pairs.csv")
rej27 = pd.read_csv("phase2/V27_REJECTION/v25_rejection_features.csv")

merged = pd.merge(pairs, v39, on="pair_id")
merged = pd.merge(merged, rej27[["pair_id","top1_score","margin","top1_corr","top1_ctx","top1_neigh","top1_grad"]], on="pair_id", how="left")
merged["loc_err"] = np.hypot(merged["gt_x"] - merged["x"], merged["gt_y"] - merged["y"])
merged = merged.fillna(0)

merged["is_correct"] = np.maximum(
    ((merged["found"]==1) & (merged["gt_found"]==1) & (merged["loc_err"]<=5.0)).astype(int),
    ((merged["found"]==0) & (merged["gt_found"]==0)).astype(int)
)

def get_strat(row):
    if row["gt_found"] == 0: return "absent"
    elif row["set_type"] == "SetA": return "A_present"
    else: return "B_present"
merged["strat_label"] = merged.apply(get_strat, axis=1)

features = ["score", "top1_score", "margin", "top1_corr", "top1_ctx", "top1_neigh", "top1_grad"]
X = merged[features].values
y = merged["is_correct"].values
strat = merged["strat_label"].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_lr = np.zeros(len(y))
oof_hgb = np.zeros(len(y))

for tr, va in skf.split(X, strat):
    # LR
    sc = StandardScaler()
    X_tr = sc.fit_transform(X[tr])
    X_va = sc.transform(X[va])
    lr = LogisticRegression(C=0.1, max_iter=1000)
    lr.fit(X_tr, y[tr])
    oof_lr[va] = lr.predict_proba(X_va)[:, 1]
    
    # HGB
    hgb = HistGradientBoostingClassifier(max_depth=2, max_iter=100, l2_regularization=1.0, random_state=42)
    hgb.fit(X[tr], y[tr])
    oof_hgb[va] = hgb.predict_proba(X[va])[:, 1]

print(f"OOF LR  AUC: {roc_auc_score(y, oof_lr):.4f}, Spearman: {spearmanr(y, oof_lr)[0]:.4f}")
print(f"OOF HGB AUC: {roc_auc_score(y, oof_hgb):.4f}, Spearman: {spearmanr(y, oof_hgb)[0]:.4f}")

