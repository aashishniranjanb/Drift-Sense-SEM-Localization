import pandas as pd, numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

v39 = pd.read_csv("phase2/V39_POSE/v39_predictions.csv")
pairs = pd.read_csv("data/phase2_dev/pairs.csv")
rej27 = pd.read_csv("phase2/V27_REJECTION/v25_rejection_features.csv")

merged = pd.merge(pairs, v39, on="pair_id")
merged = pd.merge(merged, rej27[["pair_id","top1_score","margin","top1_corr","top1_ctx","top1_neigh","top1_grad"]], on="pair_id", how="left")
merged["loc_err"] = np.hypot(merged["gt_x"] - merged["x"], merged["gt_y"] - merged["y"])
merged["correct"] = np.maximum(
    ((merged["found"]==1) & (merged["gt_found"]==1) & (merged["loc_err"]<=5.0)).astype(int),
    ((merged["found"]==0) & (merged["gt_found"]==0)).astype(int)
)

best_auc = 0
best_sp = 0
best_w = None
results = []

# Random search 10,000 combinations
np.random.seed(42)
for i in range(10000):
    w_score = np.random.uniform(0.6, 1.0)
    w_top1 = np.random.uniform(0, 0.2)
    w_corr = np.random.uniform(0, 0.2)
    w_ctx = np.random.uniform(0, 0.2)
    w_neigh = np.random.uniform(0, 0.2)
    w_grad = np.random.uniform(0, 0.2)
    
    # Normalize
    total = w_score + w_top1 + w_corr + w_ctx + w_neigh + w_grad
    w = [w_score/total, w_top1/total, w_corr/total, w_ctx/total, w_neigh/total, w_grad/total]
    
    new_scores = (w[0] * merged["score"] + 
                  w[1] * merged["top1_score"] + 
                  w[2] * merged["top1_corr"] + 
                  w[3] * merged["top1_ctx"] + 
                  w[4] * merged["top1_neigh"] + 
                  w[5] * merged["top1_grad"])
    
    sp = spearmanr(merged["correct"], new_scores)[0]
    auc = roc_auc_score(merged["correct"], new_scores)
    
    if sp > best_sp:
        best_sp = sp
        best_auc = auc
        best_w = w
        
    results.append((sp, auc, w))

print(f"Best Spearman: {best_sp:.4f} (AUC: {best_auc:.4f})")
print(f"Weights: score={best_w[0]:.3f}, top1={best_w[1]:.3f}, corr={best_w[2]:.3f}, ctx={best_w[3]:.3f}, neigh={best_w[4]:.3f}, grad={best_w[5]:.3f}")

# Sort by AUC
results.sort(key=lambda x: -x[1])
best_auc_res = results[0]
print(f"\nBest AUC: {best_auc_res[1]:.4f} (Spearman: {best_auc_res[0]:.4f})")
bw = best_auc_res[2]
print(f"Weights: score={bw[0]:.3f}, top1={bw[1]:.3f}, corr={bw[2]:.3f}, ctx={bw[3]:.3f}, neigh={bw[4]:.3f}, grad={bw[5]:.3f}")

