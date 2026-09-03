import pandas as pd, numpy as np
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

# The key insight: score is nearly binary (found=1 always gets high score like ~0.9-0.96)
# Only the 2 wrong found=1 cases (SetC pairs) have slightly lower scores
# The real calibration opportunity is in the ordering within found=1 and found=0 subsets

# Test: use top1_score as calibration signal instead of score
# top1_score has much more variance and spread (0.13-0.98)
# This is the REAL confidence, before binarization

# What happens if we replace found=0 scores (0.0) with actual presence confidence?
# The false negatives (found=0 but gt_found=1) get low scores; correct absent get 0
# But AUC is already 0.82 because of the binary gap

# The improvement track: use top1_score (or a linear combination) as the actual score
# This would differentiate the 2 wrong found=1 from the 76 correct found=1
# And differentiate among found=0 cases

# Build the combined score using top1_score + margin
for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    merged["combo"] = (1 - alpha) * merged["score"] + alpha * merged["top1_score"]
    auc = roc_auc_score(merged["correct"], merged["combo"])
    sp = spearmanr(merged["correct"], merged["combo"])[0]
    print("alpha=%.1f combo: AUC=%.4f Spearman=%.4f" % (alpha, auc, sp))
