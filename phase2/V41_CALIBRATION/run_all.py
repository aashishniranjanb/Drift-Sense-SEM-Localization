import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from scipy.stats import spearmanr
import os, subprocess, sys, zipfile

# 1. Setup Directories
os.makedirs("phase2/V41_CALIBRATION/FINAL", exist_ok=True)
os.makedirs("phase2/V41_CALIBRATION/reports", exist_ok=True)
os.makedirs("phase2/V41_CALIBRATION/FINAL_CLEANROOM", exist_ok=True)

# 2. Load Data
v39 = pd.read_csv("phase2/V39_POSE/v39_predictions.csv")
pairs = pd.read_csv("data/phase2_dev/pairs.csv")
rej27 = pd.read_csv("phase2/V27_REJECTION/v25_rejection_features.csv")

merged = pd.merge(pairs, v39, on="pair_id")
merged = pd.merge(merged, rej27[["pair_id","top1_score","margin","top1_corr"]], on="pair_id", how="left")
merged["loc_err"] = np.hypot(merged["gt_x"] - merged["x"], merged["gt_y"] - merged["y"])
merged["correct_present"] = ((merged["found"]==1) & (merged["gt_found"]==1) & (merged["loc_err"]<=5.0)).astype(int)
merged["correct_absent"] = ((merged["found"]==0) & (merged["gt_found"]==0)).astype(int)
merged["is_correct"] = np.maximum(merged["correct_present"], merged["correct_absent"])
merged = merged.fillna(0)

# 3. Apply Heuristic Calibration
# new_score = 0.90 * raw_score + 0.05 * top1_score + 0.05 * top1_corr
cal_scores = 0.90 * merged["score"] + 0.05 * merged["top1_score"] + 0.05 * merged["top1_corr"]
merged["cal_score"] = cal_scores

# 4. Compute Metrics
auc = roc_auc_score(merged["is_correct"], cal_scores)
sp = spearmanr(merged["is_correct"], cal_scores)[0]
brier = brier_score_loss(merged["is_correct"], cal_scores)
ll = log_loss(merged["is_correct"], cal_scores)

base_auc = roc_auc_score(merged["is_correct"], merged["score"])
base_sp = spearmanr(merged["is_correct"], merged["score"])[0]

# 5. Failure Analysis
wrong = merged[merged["is_correct"] == 0]
overconfident = wrong.sort_values("cal_score", ascending=False).head(20)

correct = merged[merged["is_correct"] == 1]
# For underconfident, we look at the ones with the lowest score
underconfident = correct.sort_values("cal_score", ascending=True).head(20)

with open("phase2/V41_CALIBRATION/reports/calibration_report.md", "w") as f:
    f.write("# V41 Calibration Report\n\n")
    f.write(f"**Strategy**: Residual Mix (`0.90 * score + 0.05 * top1_score + 0.05 * top1_corr`)\n\n")
    f.write("## Metrics\n")
    f.write(f"- **AUC**: {auc:.4f} (Baseline: {base_auc:.4f} | **+{auc-base_auc:.4f}**)\n")
    f.write(f"- **Spearman**: {sp:.4f} (Baseline: {base_sp:.4f} | **+{sp-base_sp:.4f}**)\n")
    f.write(f"- **Brier Score**: {brier:.4f}\n")
    f.write(f"- **Log Loss**: {ll:.4f}\n\n")
    
    f.write("## Top 20 Overconfident Wrong\n")
    cols = ["pair_id", "found", "gt_found", "cal_score", "score", "top1_score", "top1_corr", "loc_err"]
    f.write(overconfident[cols].to_csv(index=False, sep="|"))
    f.write("\n\n")
    
    f.write("## Top 20 Underconfident Correct\n")
    f.write(underconfident[cols].to_csv(index=False, sep="|"))
    f.write("\n")

# 6. Build Final Artifacts
final_preds = v39.copy()
final_preds["score"] = cal_scores
final_csv = "phase2/V41_CALIBRATION/FINAL/v41_predictions.csv"
final_preds.to_csv(final_csv, index=False)

# Clean room copy
cleanroom_csv = "phase2/V41_CALIBRATION/FINAL_CLEANROOM/predictions.csv"
final_preds.to_csv(cleanroom_csv, index=False)

# Zip it
with zipfile.ZipFile("phase2/V41_CALIBRATION/FINAL_CLEANROOM/submission.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(cleanroom_csv, arcname="predictions.csv")

print("V41 Calibration Complete.")
print(f"Final AUC: {auc:.4f} (+{auc-base_auc:.4f})")
print(f"Final Spearman: {sp:.4f} (+{sp-base_sp:.4f})")
print(f"Saved artifacts to phase2/V41_CALIBRATION/FINAL_CLEANROOM")

# 7. Run Official Benchmark
subprocess.run([sys.executable, "phase2/benchmark_phase2.py", 
                "--input-csv", "data/phase2_dev/pairs.csv",
                "--predictions-csv", final_csv])
