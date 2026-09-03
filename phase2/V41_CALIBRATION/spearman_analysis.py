import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# The Spearman in the official benchmark measures ordering of score vs 
# localization success. V39 score=0.0 for found=0 (correct absent) so the
# metric computes differently. We need to understand exactly what Spearman
# is being measured in the official benchmark

v39 = pd.read_csv("phase2/V39_POSE/v39_predictions.csv")
pairs = pd.read_csv("data/phase2_dev/pairs.csv")
rej27 = pd.read_csv("phase2/V27_REJECTION/v25_rejection_features.csv")

merged = pd.merge(pairs, v39, on="pair_id")
merged = pd.merge(merged, rej27[["pair_id","top1_score","margin","top1_corr","top1_ctx","top1_neigh","top1_grad","mode_strong"]], on="pair_id", how="left")
merged["loc_err"] = np.hypot(merged["gt_x"] - merged["x"], merged["gt_y"] - merged["y"])

# What does official Spearman check? It checks score vs correctness ordering
# For SetC (absent): correct=1 when found=0. Score=0.0. 
# For SetA/B (present): correct=1 when found=1 AND loc<=5px. Score=pres_score
# The Spearman is between score and binary-correct

# The trick: our calibrator for found=1 cases slightly hurts the monotonicity
# because we expose different score range that doesn't perfectly align with rank order
# Temperature scaling won't help AUC but might preserve Spearman better

# Key insight: The official metric is Spearman rho
# V39 Spearman = 0.5995 (baseline)
# V41 evidence-logistic Spearman = 0.5858 (worse)
# Temperature scaling = same AUC but same Spearman since monotonic

# The safest calibration is TEMPERATURE SCALING because:
# 1. Monotonic transformation - CANNOT hurt AUC
# 2. Preserves rank ordering - CANNOT hurt Spearman
# 3. Changes probability values (calibration quality) but not ordering

# Let us search for best temperature that improves calibration probability alignment
def sigmoid(x): return 1 / (1 + np.exp(-x))

eps = 1e-6
s_raw = np.clip(merged["score"].values, eps, 1 - eps)

# For found=0 pairs, score is 0.0 which becomes -inf in logit space
# We need to handle these separately
found_mask = merged["found"].values == 1
print("Pairs with found=1:", found_mask.sum())

# Test temperature on found=1 pairs only
s_found = s_raw[found_mask]
logit_found = np.log(s_found / (1 - s_found))
print(f"Logit range for found=1: {logit_found.min():.3f} to {logit_found.max():.3f}")

# This is exactly what temperature scaling does - maps logit -> logit/T
# T > 1 sharpens probabilities toward 0.5 (less confident)
# T < 1 sharpens toward extremes (more confident)

# The issue: ALL found=1 scores are already very high (0.877-0.964)
# Temperature < 1 pushes them closer to 1.0 (more confident)
# Temperature > 1 pulls them toward 0.5 (less confident, more calibrated)

# Test what the official calibration score measures:
# It says Spearman rank correlation between predicted score and localization success

# Optimal strategy for the official Spearman metric:
# Keep found=0 as 0.0 (or small epsilon), keep found=1 as high but correctly ordered
# The top1_score within found=1 subset might have better monotonic ordering

found_pairs = merged[found_mask]
gt_found_and_present = ((found_pairs["gt_found"]==1) & (found_pairs["loc_err"]<=5.0)).astype(int)
print("\nWithin found=1 pairs:")
print(f"  top1_score Spearman vs correctness: {spearmanr(found_pairs['top1_score'], gt_found_and_present)[0]:.4f}")
print(f"  score Spearman vs correctness:      {spearmanr(found_pairs['score'], gt_found_and_present)[0]:.4f}")
print(f"  margin Spearman vs correctness:     {spearmanr(found_pairs['margin'], gt_found_and_present)[0]:.4f}")
print(f"  top1_corr Spearman vs correctness:  {spearmanr(found_pairs['top1_corr'], gt_found_and_present)[0]:.4f}")
print(f"  top1_ctx Spearman vs correctness:   {spearmanr(found_pairs['top1_ctx'], gt_found_and_present)[0]:.4f}")

# Among ALL 180 pairs (with absent=0.0):
print("\nAmong all 180:")
all_correct = np.maximum(
    ((merged["found"]==1) & (merged["gt_found"]==1) & (merged["loc_err"]<=5.0)).astype(int),
    ((merged["found"]==0) & (merged["gt_found"]==0)).astype(int)
).values

for label, scores in [("score", s_raw), ("top1_score", merged["top1_score"].fillna(0).values),
                       ("margin", merged["margin"].fillna(0).values),
                       ("0.9*score+0.1*top1", 0.9*s_raw + 0.1*merged["top1_score"].fillna(0).values),
                       ("score+0.1*margin", s_raw + 0.1*merged["margin"].fillna(0).values)]:
    sp = spearmanr(all_correct, scores)[0]
    auc = roc_auc_score(all_correct, scores)
    print(f"  {label:30s}: Spearman={sp:.4f}, AUC={auc:.4f}")
