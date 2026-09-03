import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

v39 = pd.read_csv("phase2/V39_POSE/v39_predictions.csv")
pairs = pd.read_csv("data/phase2_dev/pairs.csv")
rej27 = pd.read_csv("phase2/V27_REJECTION/v25_rejection_features.csv")

merged = pd.merge(pairs, v39, on="pair_id")
merged = pd.merge(merged, rej27[["pair_id","top1_score","margin","top1_corr","top1_ctx","top1_neigh","top1_grad","mode_strong"]], on="pair_id", how="left")

merged["loc_err"] = np.hypot(merged["gt_x"] - merged["x"], merged["gt_y"] - merged["y"])
merged["correct_present"] = ((merged["found"]==1) & (merged["gt_found"]==1) & (merged["loc_err"]<=5.0)).astype(int)
merged["correct_absent"] = ((merged["found"]==0) & (merged["gt_found"]==0)).astype(int)
merged["correct"] = np.maximum(merged["correct_present"], merged["correct_absent"])

print("Total pairs:", len(merged))
print("Correct predictions:", merged["correct"].sum())
print("Correct present (found=1, loc<=5px):", merged["correct_present"].sum())
print("Correct absent (found=0, gt_found=0):", merged["correct_absent"].sum())
print()

correct_mask = merged["correct"]==1
wrong_mask = merged["correct"]==0
sc_c = merged.loc[correct_mask, "score"]
sc_w = merged.loc[wrong_mask, "score"]
print(f"Correct (n={correct_mask.sum()}): score mean={sc_c.mean():.4f}, median={sc_c.median():.4f}, min={sc_c.min():.4f}")
print(f"Wrong   (n={wrong_mask.sum()}): score mean={sc_w.mean():.4f}, median={sc_w.median():.4f}, min={sc_w.min():.4f}")
print()

auc_all = roc_auc_score(merged["correct"], merged["score"])
sp_all = spearmanr(merged["correct"], merged["score"])[0]
found_pairs = merged[merged["found"]==1]
auc_found = roc_auc_score(found_pairs["correct_present"], found_pairs["score"])
sp_found = spearmanr(found_pairs["correct_present"], found_pairs["score"])[0]
print(f"AUC on all 180: {auc_all:.4f}")
print(f"Spearman on all 180: {sp_all:.4f}")
print(f"AUC on found=1 (78 pairs): {auc_found:.4f}")
print(f"Spearman on found=1 (78 pairs): {sp_found:.4f}")
