import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import subprocess, sys, warnings
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
merged = merged.fillna(0)

# Grid search all combinations
print("Grid search over alpha:")
best_auc = 0
best_alpha = 0.1
for alpha in np.arange(0.00, 0.31, 0.01):
    combo = (1 - alpha) * merged["score"] + alpha * merged["top1_score"]
    combo_safe = combo.copy()
    combo_safe[merged["found"] == 0] = 0.0
    sp = spearmanr(merged["correct"], combo_safe)[0]
    auc = roc_auc_score(merged["correct"], combo_safe)
    sp_vs_raw = spearmanr(combo_safe, merged["score"])[0]
    if auc > best_auc:
        best_auc = auc
        best_sp = sp
        best_alpha = alpha
        best_sp_vs_raw = sp_vs_raw
    if alpha in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        print("  alpha=%.2f: AUC=%.4f Spearman=%.4f sp_vs_raw=%.4f" % (alpha, auc, sp, sp_vs_raw))

print("Best: alpha=%.2f AUC=%.4f Spearman=%.4f" % (best_alpha, best_auc, best_sp))

# Build final predictions with best alpha
merged["new_score"] = (1 - best_alpha) * merged["score"] + best_alpha * merged["top1_score"]
merged.loc[merged["found"] == 0, "new_score"] = 0.0

final_preds = v39.copy()
pid_to_newscore = dict(zip(merged["pair_id"], merged["new_score"]))
final_preds["score"] = final_preds["pair_id"].map(pid_to_newscore)

assert (final_preds["x"].values == v39["x"].values).all()
assert (final_preds["y"].values == v39["y"].values).all()
assert (final_preds["theta"].values == v39["theta"].values).all()
assert (final_preds["scale"].values == v39["scale"].values).all()
assert (final_preds["found"].values == v39["found"].values).all()
print("SAFETY: All frozen fields identical - CONFIRMED")

out_path = "phase2/V41_CALIBRATION/experiments/v41_best_mix_predictions.csv"
final_preds.to_csv(out_path, index=False)
print("Saved to", out_path)

result = subprocess.run([sys.executable, "phase2/benchmark_phase2.py",
    "--input-csv", "data/phase2_dev/pairs.csv",
    "--predictions-csv", out_path], capture_output=True, text=True)
print(result.stdout)
