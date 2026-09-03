import pandas as pd, numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# The re-scoring hurts because:
# - REJECTION_SUCCESS (38 correct absent) currently has score=0.0 -> ranks last
# - PRESENCE_FALSE_NEGATIVE (64 wrong) also has score=0.0 -> same rank
# - When we add non-zero to found=0, they all move up, and since
#   the FN (64) are tied with REJECTION (38), adding non-zero to found=0
#   actually HURTS because we're re-ordering WITHIN the found=0 group
#   in a way that doesn't help the global Spearman

# The actual Spearman is dominated by: found=1 (score 0.87-0.96) vs found=0 (score 0.0)
# The found=1 group is always ranked high regardless of correctness (76/78 correct)
# The 2 wrong found=1 slightly depress Spearman  
# The 102 found=0 all have score=0 and we can't distinguish them
# Spearman = 0.5995 is essentially the ceiling when found ordering = score ordering

# What is the THEORETICAL MAXIMUM Spearman without changing found?
# Best case: 76 correct found=1 all ranked 1 (score 1.0)
# 2 wrong found=1 ranked lowest (score 0.0)
# 38 correct absent (found=0) ranked just above 2 wrong found=1
# 64 FN (wrong found=0) ranked last

v39 = pd.read_csv("phase2/V39_POSE/v39_predictions.csv")
pairs = pd.read_csv("data/phase2_dev/pairs.csv")
rej27 = pd.read_csv("phase2/V27_REJECTION/v25_rejection_features.csv")
merged = pd.merge(pairs, v39, on="pair_id")
merged = pd.merge(merged, rej27[["pair_id","top1_score","margin","top1_corr"]], on="pair_id", how="left")
merged["loc_err"] = np.hypot(merged["gt_x"] - merged["x"], merged["gt_y"] - merged["y"])
merged = merged.fillna(0)

def get_failure_mode(row):
    found = row["found"]; gt_found = row["gt_found"]; loc_err = row["loc_err"]
    if found == 0 and gt_found == 0: return "REJECTION_SUCCESS"
    elif found == 0 and gt_found == 1: return "PRESENCE_FALSE_NEGATIVE"
    elif found == 1 and gt_found == 0: return "ABSENCE_FALSE_POSITIVE"
    elif found == 1 and gt_found == 1:
        if loc_err <= 1.0: return "SUBPIXEL_SUCCESS"
        elif loc_err <= 5.0: return "IN_BOUNDS_SUCCESS"
        else: return "PERIODIC_REPLICA"
    return "UNKNOWN"

merged["failure_mode"] = merged.apply(get_failure_mode, axis=1)
correct_modes = {"SUBPIXEL_SUCCESS", "IN_BOUNDS_SUCCESS", "REJECTION_SUCCESS"}
merged["is_correct"] = merged["failure_mode"].apply(lambda x: 1 if x in correct_modes else 0)

# Theoretical oracle: ideal score assignment
oracle_scores = np.zeros(len(merged))
for i, row in merged.iterrows():
    fm = row["failure_mode"]
    if fm in ["SUBPIXEL_SUCCESS", "IN_BOUNDS_SUCCESS"]:
        oracle_scores[i] = 1.0
    elif fm == "REJECTION_SUCCESS":
        oracle_scores[i] = 0.3  # moderate-high for correct absence
    elif fm == "PRESENCE_FALSE_NEGATIVE":
        oracle_scores[i] = 0.1  # low for missed detection
    elif fm in ["ABSENCE_FALSE_POSITIVE", "PERIODIC_REPLICA"]:
        oracle_scores[i] = 0.0  # lowest for wrong found

sp_oracle = spearmanr(oracle_scores, merged["is_correct"])[0]
print(f"Oracle Spearman (ideal ordering): {sp_oracle:.4f}")
print(f"Baseline Spearman: {spearmanr(merged['score'], merged['is_correct'])[0]:.4f}")

# Can we improve within found=1 only?
# Wrong found=1: 2 SetC false positives (pair_140, pair_159)
# Give them lower score
improved_scores = merged["score"].copy()
# Lower the 2 wrong found=1 scores
fp_mask = merged["failure_mode"] == "ABSENCE_FALSE_POSITIVE"
improved_scores[fp_mask] = improved_scores[fp_mask] * 0.5  # halve their score
sp_imp = spearmanr(improved_scores, merged["is_correct"])[0]
print(f"If we halve score of 2 wrong found=1: Spearman={sp_imp:.4f}")

# What if we separate FN from REJECTION_SUCCESS using top1_score?
# For found=0: REJECTION (correct) should have low top1_score
# FN (wrong) might have higher top1_score (just under threshold)
found_0 = merged[merged["found"]==0]
print()
print("Within found=0:")
print("REJECTION_SUCCESS top1_score: mean=%.4f, median=%.4f" % (
    found_0[found_0["failure_mode"]=="REJECTION_SUCCESS"]["top1_score"].mean(),
    found_0[found_0["failure_mode"]=="REJECTION_SUCCESS"]["top1_score"].median()))
print("PRESENCE_FN top1_score: mean=%.4f, median=%.4f" % (
    found_0[found_0["failure_mode"]=="PRESENCE_FALSE_NEGATIVE"]["top1_score"].mean(),
    found_0[found_0["failure_mode"]=="PRESENCE_FALSE_NEGATIVE"]["top1_score"].median()))
