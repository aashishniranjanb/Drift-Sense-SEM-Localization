import pandas as pd, numpy as np
import scipy.stats as stats

v39 = pd.read_csv("phase2/V39_POSE/v39_predictions.csv")
pairs = pd.read_csv("data/phase2_dev/pairs.csv")
rej27 = pd.read_csv("phase2/V27_REJECTION/v25_rejection_features.csv")

merged = pd.merge(pairs, v39, on="pair_id")
merged = pd.merge(merged, rej27[["pair_id","top1_score","margin","top1_corr","top1_ctx","top1_neigh","top1_grad","mode_strong"]], on="pair_id", how="left")
merged["loc_err"] = np.hypot(merged["gt_x"] - merged["x"], merged["gt_y"] - merged["y"])
merged = merged.fillna(0)

def get_failure_mode(row):
    found = row["found"]; gt_found = row["gt_found"]; loc_err = row["loc_err"]
    if found == 0 and gt_found == 0: return "REJECTION_SUCCESS"
    elif found == 0 and gt_found == 1: return "PRESENCE_FALSE_NEGATIVE"
    return "OTHER"

merged["failure_mode"] = merged.apply(get_failure_mode, axis=1)
found_0 = merged[merged["found"]==0]

print("Feature means for found=0:")
features = ["top1_score", "margin", "top1_corr", "top1_ctx", "top1_neigh", "top1_grad", "mode_strong"]

for f in features:
    mean_rej = found_0[found_0["failure_mode"]=="REJECTION_SUCCESS"][f].mean()
    mean_fn = found_0[found_0["failure_mode"]=="PRESENCE_FALSE_NEGATIVE"][f].mean()
    t, p = stats.ttest_ind(found_0[found_0["failure_mode"]=="REJECTION_SUCCESS"][f], 
                          found_0[found_0["failure_mode"]=="PRESENCE_FALSE_NEGATIVE"][f])
    print(f"{f:12s} - RejSuccess (correct): {mean_rej:.4f} | FN (wrong): {mean_fn:.4f} | p-val: {p:.4f}")

# Look for correlation with correctness within found=0
correct_in_found_0 = (found_0["failure_mode"] == "REJECTION_SUCCESS").astype(int)
print("\nSpearman with correctness inside found=0:")
for f in features:
    sp = stats.spearmanr(found_0[f], correct_in_found_0)[0]
    print(f"{f:12s}: {sp:.4f}")

