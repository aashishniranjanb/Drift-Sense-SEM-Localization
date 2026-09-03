import pandas as pd, numpy as np
from scipy.stats import spearmanr
import subprocess, sys, warnings
warnings.filterwarnings("ignore")

# KEY INSIGHT: Spearman is on ALL 180 pairs, but score for found=0 is 0.0
# failure_modes: SUBPIXEL_SUCCESS / IN_BOUNDS_SUCCESS / REJECTION_SUCCESS = correct
# others = wrong
# REJECTION_SUCCESS = found=0 AND gt_found=0 (correct absence) -> score=0.0
# The correct absent pairs get score=0.0, and wrong FN also get 0.0
# So 64 FN (wrong, score=0) and 38 Rejection Success (correct, score=0) are tied
# This is WHY Spearman is stuck around 0.60 - we cannot differentiate them!

# The ONLY way to improve Spearman is:
# Option A: Give correct absence (REJECTION_SUCCESS) score > found=0 wrong cases
#   -> We need to re-score found=0 pairs based on absence confidence
# Option B: Fix the 2 wrong found=1 to be lower scores (minor)

# For option A: use presence_score to differentiate WITHIN found=0
# The presence model produces pres_score < 0.843 for found=0 pairs
# Higher pres_score among found=0 = more ambiguous = likely FN (absent reported)
# Lower pres_score among found=0 = more clearly absent = REJECTION_SUCCESS

# For found=0 pairs, current score = 0.0 for ALL
# If we set score = 1 - top1_score for found=0 pairs:
#   - Clear absence (low top1_score) -> higher score -> higher rank -> correct
#   - Ambiguous FN (high top1_score pushed below threshold) -> lower score

v39 = pd.read_csv("phase2/V39_POSE/v39_predictions.csv")
pairs = pd.read_csv("data/phase2_dev/pairs.csv")
rej27 = pd.read_csv("phase2/V27_REJECTION/v25_rejection_features.csv")

merged = pd.merge(pairs, v39, on="pair_id")
merged = pd.merge(merged, rej27[["pair_id","top1_score","margin","top1_corr","top1_ctx","top1_neigh","top1_grad","mode_strong"]], on="pair_id", how="left")
merged["loc_err"] = np.hypot(merged["gt_x"] - merged["x"], merged["gt_y"] - merged["y"])
merged = merged.fillna(0)

# Compute failure modes manually (what benchmark does)
def get_failure_mode(row):
    found = row["found"]
    gt_found = row["gt_found"]
    loc_err = row["loc_err"]
    if found == 0 and gt_found == 0:
        return "REJECTION_SUCCESS"
    elif found == 0 and gt_found == 1:
        return "PRESENCE_FALSE_NEGATIVE"
    elif found == 1 and gt_found == 0:
        return "ABSENCE_FALSE_POSITIVE"
    elif found == 1 and gt_found == 1:
        if loc_err <= 1.0:
            return "SUBPIXEL_SUCCESS"
        elif loc_err <= 5.0:
            return "IN_BOUNDS_SUCCESS"
        else:
            return "PERIODIC_REPLICA"
    return "UNKNOWN"

merged["failure_mode"] = merged.apply(get_failure_mode, axis=1)
correct_modes = {"SUBPIXEL_SUCCESS", "IN_BOUNDS_SUCCESS", "REJECTION_SUCCESS"}
merged["is_correct"] = merged["failure_mode"].apply(lambda x: 1 if x in correct_modes else 0)

print("Failure mode distribution:")
print(merged["failure_mode"].value_counts().to_string())
print()

# Baseline Spearman
sp_base = spearmanr(merged["score"], merged["is_correct"])[0]
print(f"Baseline Spearman: {sp_base:.4f}")
print()

# Now try: for found=0 pairs, use (1 - top1_score) as confidence
# This says: if I rejected you with low confidence (high top1_score), you are probably a FN
# If I rejected you with high confidence (low top1_score), you are probably correct absent

print("Testing found=0 re-scoring strategies:")
for scale in [0.1, 0.2, 0.3, 0.4, 0.5]:
    test_scores = merged["score"].copy()
    # For found=0 pairs: assign score = scale * (1 - top1_score)
    mask_0 = merged["found"] == 0
    test_scores[mask_0] = scale * (1 - merged.loc[mask_0, "top1_score"])
    sp = spearmanr(test_scores, merged["is_correct"])[0]
    print(f"  scale={scale:.1f}: Spearman={sp:.4f} (delta={sp - sp_base:+.4f})")

print()
# Also try: just give correct absent a small bonus using absence certainty
for bonus in [0.05, 0.10, 0.15, 0.20, 0.30]:
    test_scores = merged["score"].copy()
    # For found=0: score = bonus * (1 - top1_score) 
    # This monotonically maps: low top1 -> high score -> correct absent
    mask_0 = merged["found"] == 0
    absence_confidence = bonus * np.clip(1 - merged.loc[mask_0, "top1_score"], 0, 1)
    test_scores[mask_0] = absence_confidence
    sp = spearmanr(test_scores, merged["is_correct"])[0]
    print(f"  bonus={bonus:.2f}: Spearman={sp:.4f} (delta={sp - sp_base:+.4f})")
