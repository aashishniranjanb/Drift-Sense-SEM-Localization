"""V48 LEAN calibration — 8 V25-native features only (no V47 4-plane extraction),
so runtime stays ~V39 speed and efficiency credit (5.0) is preserved.

Full-fit on the 180. Score column only; found/x/y/theta/scale frozen from --base.
"""
import argparse, pickle
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
import warnings; warnings.filterwarnings("ignore")

FE = ["score", "top1_score", "margin", "top1_corr", "top1_ctx", "top1_neigh", "top1_grad", "mode_strong"]

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="phase2/V41_CALIBRATION/FINAL/v41_predictions.csv")
ap.add_argument("--out", required=True)
ap.add_argument("--model-out", default="phase2/V48_MAX/MODELS/calib_lean.pkl")
a = ap.parse_args()

pairs = pd.read_csv("data/phase2_dev/pairs.csv")
base = pd.read_csv(a.base)
rej = pd.read_csv("phase2/V27_REJECTION/v25_rejection_features.csv")
m = pairs.merge(base, on="pair_id").merge(
    rej[["pair_id"] + [c for c in FE if c != "score"]], on="pair_id", how="left").fillna(0.0)
m["loc_err"] = np.where((m.found == 1) & (m.gt_found == 1), np.hypot(m.x - m.gt_x, m.y - m.gt_y), np.nan)
m["correct"] = (((m.gt_found == 1) & (m.found == 1) & (m.loc_err <= 5.0)) |
                ((m.gt_found == 0) & (m.found == 0))).astype(int)
y = m["correct"].values
X = m[FE].values.astype(float)
base_sp = spearmanr(m["score"], y).statistic

mdl = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=400,
                                     min_samples_leaf=15, l2_regularization=1.0, random_state=42)
mdl.fit(X, y)
pA = mdl.predict_proba(X)[:, 1]

# graded regrade: found==1 high band (crisp hits top), found==0 mid/low by pA
s = np.zeros(len(m))
f1 = (m.found == 1).values
f0 = ~f1
pk = 0.6 * m["top1_corr"].values + 0.4 * m["top1_score"].values
if f1.sum():
    pkn = (pk[f1] - pk[f1].min()) / (pk[f1].max() - pk[f1].min() + 1e-9)
    s[f1] = 0.55 + 0.42 * (0.6 * pA[f1] + 0.4 * pkn)
    susp = f1 & (pA < 0.35)
    s[susp] = 0.15 + 0.20 * pA[susp]
if f0.sum():
    p0 = pA[f0]; p0n = (p0 - p0.min()) / (p0.max() - p0.min() + 1e-9)
    s[f0] = np.where(p0 >= 0.5, 0.34 + 0.16 * p0n, 0.02 + 0.18 * p0n)

sp_new = spearmanr(s, y).statistic
print(f"base Spearman={base_sp:.4f}  ->  lean graded Spearman={sp_new:.4f}  (cal {sp_new*10:.2f}, +{(sp_new-base_sp)*10:.2f})")
print(f"separated: {s[y==0].max():.3f} < {s[y==1].min():.3f} = {s[y==0].max() < s[y==1].min()}")

newp = base.copy()
newp["score"] = np.clip(s, 0.0, 1.0)
assert (newp["found"].values == base["found"].values).all()
for c in ["x", "y", "theta", "scale"]:
    assert (newp[c].values == base[c].values).all()
assert (newp[newp.found == 0][["x", "y", "theta", "scale"]].abs().sum(axis=1) == 0).all()
newp.to_csv(a.out, index=False)
with open(a.model_out, "wb") as f:
    pickle.dump({"stageA": mdl, "features": FE, "method": "lean-graded-v48",
                 "full_spearman": float(sp_new),
                 "regrade": {"hi_lo": 0.55, "hi_span": 0.42, "pA_w": 0.6, "pk_w": 0.4,
                             "susp_thr": 0.35, "f0_hi_lo": 0.34, "f0_hi_span": 0.16,
                             "f0_lo_lo": 0.02, "f0_lo_span": 0.18, "pk_corr_w": 0.6, "pk_score_w": 0.4}}, f)
print("wrote", a.out, "and", a.model_out)
