import pandas as pd, numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

v39 = pd.read_csv("phase2/V39_POSE/v39_predictions.csv")
pairs = pd.read_csv("data/phase2_dev/pairs.csv")
rej27 = pd.read_csv("phase2/V27_REJECTION/v25_rejection_features.csv")

merged = pd.merge(pairs, v39, on="pair_id")
merged = pd.merge(merged, rej27[["pair_id","top1_score","margin","top1_corr","top1_ctx","top1_neigh","top1_grad","mode_strong"]], on="pair_id", how="left")
merged["loc_err"] = np.hypot(merged["gt_x"] - merged["x"], merged["gt_y"] - merged["y"])
merged["correct"] = np.maximum(
    ((merged["found"]==1) & (merged["gt_found"]==1) & (merged["loc_err"]<=5.0)).astype(int),
    ((merged["found"]==0) & (merged["gt_found"]==0)).astype(int)
)

best_auc = 0
best_sp = 0
best_w = None

# Grid search linear combinations of evidence to form the score
# Score = w0*score + w1*top1_score + w2*top1_corr + w3*top1_ctx
for w1 in np.arange(0.0, 0.4, 0.05):
    for w2 in np.arange(0.0, 0.4, 0.05):
        for w3 in np.arange(0.0, 0.4, 0.05):
            w0 = 1.0 - w1 - w2 - w3
            if w0 < 0.4: continue
            
            new_scores = w0 * merged["score"] + w1 * merged["top1_score"] + w2 * merged["top1_corr"] + w3 * merged["top1_ctx"]
            
            sp = spearmanr(merged["correct"], new_scores)[0]
            auc = roc_auc_score(merged["correct"], new_scores)
            
            # We want to maximize Spearman first, then AUC
            if sp > best_sp:
                best_sp = sp
                best_auc = auc
                best_w = (w0, w1, w2, w3)

print(f"Best Weights: score={best_w[0]:.2f}, top1={best_w[1]:.2f}, corr={best_w[2]:.2f}, ctx={best_w[3]:.2f}")
print(f"AUC: {best_auc:.4f}, Spearman: {best_sp:.4f}")

