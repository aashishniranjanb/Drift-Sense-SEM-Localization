import pandas as pd, numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import subprocess, sys

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

merged = merged.fillna(0)
# Final formula
new_scores = 0.90 * merged["score"] + 0.05 * merged["top1_score"] + 0.05 * merged["top1_corr"]

auc = roc_auc_score(merged["correct"], new_scores)
sp = spearmanr(merged["correct"], new_scores)[0]
print(f"Final Formula AUC: {auc:.4f}, Spearman: {sp:.4f}")

# Produce FINAL predictions
final_preds = v39.copy()
final_preds["score"] = new_scores

out_path = "phase2/V41_CALIBRATION/FINAL/v41_predictions.csv"
final_preds.to_csv(out_path, index=False)

# Run Official Benchmark
result = subprocess.run([sys.executable, "phase2/benchmark_phase2.py",
    "--input-csv", "data/phase2_dev/pairs.csv",
    "--predictions-csv", out_path], capture_output=True, text=True)

print("\n--- OFFICIAL BENCHMARK ---")
for line in result.stdout.split("\n"):
    if "Spearman" in line or "AUC" in line or "TAXONOMY" in line:
        print(line)

