import pandas as pd
import numpy as np

v39 = pd.read_csv("phase2/V39_POSE/v39_predictions.csv")
pairs = pd.read_csv("data/phase2_dev/pairs.csv")
rej27 = pd.read_csv("phase2/V27_REJECTION/v25_rejection_features.csv")

merged = pd.merge(pairs, v39, on="pair_id")
merged = pd.merge(merged, rej27[["pair_id","top1_score","margin","top1_corr","top1_ctx","top1_neigh","top1_grad","mode_strong"]], on="pair_id", how="left")
merged["loc_err"] = np.hypot(merged["gt_x"] - merged["x"], merged["gt_y"] - merged["y"])
merged["correct_present"] = ((merged["found"]==1) & (merged["gt_found"]==1) & (merged["loc_err"]<=5.0)).astype(int)

found_1 = merged[merged["found"]==1]
print("Found=1 score: mean=%.4f std=%.4f min=%.4f max=%.4f unique=%d" % (
    found_1["score"].mean(), found_1["score"].std(), found_1["score"].min(), found_1["score"].max(), len(found_1["score"].unique())))
print("top1_score (pre-binary): mean=%.4f std=%.4f min=%.4f max=%.4f" % (
    found_1["top1_score"].mean(), found_1["top1_score"].std(), found_1["top1_score"].min(), found_1["top1_score"].max()))
print("margin: mean=%.4f std=%.4f min=%.4f max=%.4f" % (
    found_1["margin"].mean(), found_1["margin"].std(), found_1["margin"].min(), found_1["margin"].max()))

fn_correct = found_1[found_1["correct_present"]==1]
fn_wrong = found_1[found_1["correct_present"]==0]
print("found=1 correct=%d, wrong=%d" % (len(fn_correct), len(fn_wrong)))
if len(fn_wrong) > 0:
    print("Wrong found=1 (periodic replicas/wrong localization):")
    print(fn_wrong[["pair_id","set_type","score","top1_score","loc_err"]].to_string())

print()
print("Correct found=1 (loc correct) - top1_score distribution:")
for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
    val = fn_correct["top1_score"].quantile(q)
    print(f"  P{int(q*100)}: {val:.4f}")
