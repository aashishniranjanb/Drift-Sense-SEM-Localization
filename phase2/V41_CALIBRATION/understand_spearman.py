import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

# Deep forensic analysis of why Spearman is stuck at 0.5995
# The official benchmark computes Spearman between score and something.
# Let us understand the 180 pairs exactly.

v39 = pd.read_csv("phase2/V39_POSE/v39_predictions.csv")
pairs = pd.read_csv("data/phase2_dev/pairs.csv")
rej27 = pd.read_csv("phase2/V27_REJECTION/v25_rejection_features.csv")

merged = pd.merge(pairs, v39, on="pair_id")
merged = pd.merge(merged, rej27[["pair_id","top1_score","margin","top1_corr","top1_ctx","top1_neigh","top1_grad","mode_strong"]], on="pair_id", how="left")
merged["loc_err"] = np.hypot(merged["gt_x"] - merged["x"], merged["gt_y"] - merged["y"])
merged = merged.fillna(0)

# The Spearman metric in the benchmark is probably score vs localization_success
# where localization_success = 1 if found=1 AND loc<=5px, OR found=0 AND gt_found=0
# (this is: correctly handling both presence and absence)

# What the Spearman ceiling is without changing found:
# Group 1: found=1, gt_found=1, loc<=5px (76 pairs) -> ideally score=1.0, rank high
# Group 2: found=1, gt_found=0 (2 pairs SetC) -> score is moderate 0.87-0.92, ideally rank LOW
# Group 3: found=0, gt_found=1 (64 pairs - FN) -> score=0.0, ideally rank=moderate
# Group 4: found=0, gt_found=0 (38 pairs) -> score=0.0, ideally rank high

# The problem: BOTH Group 3 (wrong, low priority) and Group 4 (correct) have score=0.0
# They are tied at rank 0.0, so Spearman can't distinguish them

# The 2 wrong found=1 (Group 2) are ranked ABOVE Group 4 (correct absent)
# which is the main source of calibration error

# To improve Spearman within the score constraint:
# Only option is: give the 2 wrong SetC pairs (found=1) lower scores
# AND give the 38 correct absent pairs higher scores (non-zero)
# But this would need to change the score for found=0 pairs from 0.0 to something

# Check: what if correct absent get score = ~0.5 (non-zero)?
# This breaks the convention that found=0 -> score=0.0

# Actually the official scorer might not care about this convention
# Let us check what the benchmark actually computes

# Read the benchmark code
with open("phase2/benchmark_phase2.py", "r") as f:
    bench_code = f.read()

# Find the Spearman computation
lines = bench_code.split("\n")
for i, line in enumerate(lines):
    if "spearman" in line.lower() or "rank" in line.lower():
        print(f"Line {i}: {line}")
