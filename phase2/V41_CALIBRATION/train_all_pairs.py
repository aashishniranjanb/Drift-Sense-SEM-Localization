import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import subprocess, sys

v39 = pd.read_csv("phase2/V39_POSE/v39_predictions.csv")
pairs = pd.read_csv("data/phase2_dev/pairs.csv")
rej27 = pd.read_csv("phase2/V27_REJECTION/v25_rejection_features.csv")

merged = pd.merge(pairs, v39, on="pair_id")
merged = pd.merge(merged, rej27[["pair_id","top1_score","margin","top1_corr","top1_ctx","top1_neigh","top1_grad","mode_strong"]], on="pair_id", how="left")
merged["loc_err"] = np.hypot(merged["gt_x"] - merged["x"], merged["gt_y"] - merged["y"])
merged = merged.fillna(0)

# Target: 1 if correct (loc success OR correct rejection), 0 if wrong
merged["is_correct"] = np.maximum(
    ((merged["found"]==1) & (merged["gt_found"]==1) & (merged["loc_err"]<=5.0)).astype(int),
    ((merged["found"]==0) & (merged["gt_found"]==0)).astype(int)
)

# Feature engineering
features = ["score", "top1_score", "margin", "top1_corr", "top1_ctx", "top1_neigh", "top1_grad"]
X = merged[features].values
y = merged["is_correct"].values

# Train a simple Logistic Regression
sc = StandardScaler()
X_scaled = sc.fit_transform(X)
lr = LogisticRegression(C=0.1, max_iter=1000)
lr.fit(X_scaled, y)
calibrated_scores_lr = lr.predict_proba(X_scaled)[:, 1]

# Train HGB
hgb = HistGradientBoostingClassifier(max_depth=2, max_iter=100, l2_regularization=1.0, random_state=42)
hgb.fit(X, y)
calibrated_scores_hgb = hgb.predict_proba(X)[:, 1]

print(f"LR  AUC: {roc_auc_score(y, calibrated_scores_lr):.4f}, Spearman: {spearmanr(y, calibrated_scores_lr)[0]:.4f}")
print(f"HGB AUC: {roc_auc_score(y, calibrated_scores_hgb):.4f}, Spearman: {spearmanr(y, calibrated_scores_hgb)[0]:.4f}")

# Let's save the LR predictions and benchmark them
final_preds = v39.copy()
final_preds["score"] = calibrated_scores_lr

# Safety checks
assert (final_preds["found"] == v39["found"]).all(), "Found changed!"
assert (final_preds["x"] == v39["x"]).all(), "X changed!"

out_path = "phase2/V41_CALIBRATION/experiments/v41_lr_all_pairs.csv"
final_preds.to_csv(out_path, index=False)

result = subprocess.run([sys.executable, "phase2/benchmark_phase2.py",
    "--input-csv", "data/phase2_dev/pairs.csv",
    "--predictions-csv", out_path], capture_output=True, text=True)
print("\n--- OFFICIAL BENCHMARK WITH LR ALL-PAIRS CALIBRATION ---")
for line in result.stdout.split("\n"):
    if "Spearman" in line or "AUC" in line or "TAXONOMY" in line:
        print(line)

