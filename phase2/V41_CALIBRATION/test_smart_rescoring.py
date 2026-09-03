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

print(f"Base AUC: {roc_auc_score(merged['correct'], merged['score']):.4f}")
print(f"Base Spearman: {spearmanr(merged['correct'], merged['score'])[0]:.4f}")

# Experiment: smart re-scoring
# For found=1: 0.9 * score + 0.1 * top1_score
# For found=0: 0.5 * top1_score  (since top1_score correlates +0.20 with correctness here!)

for f0_scale in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
    new_scores = merged["score"].copy()
    
    # Update found=1
    mask_1 = merged["found"] == 1
    new_scores[mask_1] = 0.9 * merged.loc[mask_1, "score"] + 0.1 * merged.loc[mask_1, "top1_score"]
    
    # Update found=0
    mask_0 = merged["found"] == 0
    new_scores[mask_0] = f0_scale * merged.loc[mask_0, "top1_score"]
    
    auc = roc_auc_score(merged["correct"], new_scores)
    sp = spearmanr(merged["correct"], new_scores)[0]
    print(f"f0_scale={f0_scale:.1f}: AUC={auc:.4f}, Spearman={sp:.4f}")

