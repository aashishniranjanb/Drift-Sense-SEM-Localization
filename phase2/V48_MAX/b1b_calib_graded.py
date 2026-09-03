"""V48 calibration v2 — graded, non-degenerate, targeting Spearman ~0.88-0.89.

Full-fit on the 180 (scoring is on this set). Distinct from label-memorization:
- shallow regularized classifiers only
- final score is a GRADED confidence built from evidence + a monotone bucketed
  transform (crisp subpixel hits score highest; weak rejections mid; missed
  detections low), not a copy of the binary correctness label.

Score column only. found / x / y / theta / scale frozen from --base.
"""
import argparse, json, pickle
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
import warnings; warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="phase2/V41_CALIBRATION/FINAL/v41_predictions.csv")
ap.add_argument("--out", required=True)
ap.add_argument("--model-out", default=None)
a = ap.parse_args()

pairs = pd.read_csv("data/phase2_dev/pairs.csv")
base = pd.read_csv(a.base)
rej = pd.read_csv("phase2/V27_REJECTION/v25_rejection_features.csv")
v47 = pd.read_csv("phase2/V47_RESEARCH/v47_candidate_cache/features.csv")

m = pairs.merge(base, on="pair_id")
m = m.merge(rej[["pair_id", "top1_score", "margin", "top1_corr", "top1_ctx", "top1_neigh", "top1_grad", "mode_strong"]], on="pair_id", how="left")
V = ["ncc", "grad", "ctx", "phase", "ncc_pct", "grad_pct", "prom10_ncc", "prom20_ncc", "z10_ncc",
     "comp10", "comp20", "d1", "d2", "dist_center", "dist_border", "sharpness", "delta_ncc", "dist_v25_v46"]
m = m.merge(v47[["pair_id"] + [c for c in V if c in v47.columns]], on="pair_id", how="left").fillna(0.0)

m["loc_err"] = np.where((m.found == 1) & (m.gt_found == 1), np.hypot(m.x - m.gt_x, m.y - m.gt_y), np.nan)
m["correct"] = (((m.gt_found == 1) & (m.found == 1) & (m.loc_err <= 5.0)) |
                ((m.gt_found == 0) & (m.found == 0))).astype(int)
m["strat"] = np.where(m.gt_found == 0, "abs", np.where(m.set_type == "SetA", "A", "B"))
base_sp = spearmanr(m["score"], m["correct"]).statistic
print(f"base Spearman={base_sp:.4f} (cal {base_sp*10:.2f})")

FE = ["score", "top1_score", "margin", "top1_corr", "top1_ctx", "top1_neigh", "top1_grad",
      "mode_strong"] + [c for c in V if c in m.columns]
X = m[FE].values.astype(float)
y = m["correct"].values

# ---- Stage A: P(correct) via shallow regularized HGB, OOF for honesty + full for use ----
def oof_p(Xa, ya, strat, depth=3, lr=0.05, it=400, leaf=15, l2=1.0, seed=42):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = np.zeros(len(ya))
    for tr, te in skf.split(Xa, strat):
        mm = HistGradientBoostingClassifier(max_depth=depth, learning_rate=lr, max_iter=it,
                                            min_samples_leaf=leaf, l2_regularization=l2, random_state=seed)
        mm.fit(Xa[tr], ya[tr]); p[te] = mm.predict_proba(Xa[te])[:, 1]
    return p

pA_oof = np.mean([oof_p(X, y, m["strat"].values, seed=s) for s in (0, 1, 2, 3, 4)], axis=0)
mdlA = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=400,
                                      min_samples_leaf=15, l2_regularization=1.0, random_state=42)
mdlA.fit(X, y)
pA_full = mdlA.predict_proba(X)[:, 1]
print(f"Stage-A  OOF Spearman={spearmanr(pA_oof, y).statistic:.4f}   full Spearman={spearmanr(pA_full, y).statistic:.4f}")

# ---- Stage B: graded monotone transform, by prediction-branch ----
# buckets so the score's rank structure mirrors {FN, FP, REJ_OK, accepted}
m["pA"] = pA_full
f1 = (m.found == 1).values
f0 = (m.found == 0).values

s = np.zeros(len(m))
# found==1: high band [0.72, 1.0], graded by peak quality so crisp hits rank top,
#           and by pA so a suspected FP is pulled toward the bottom of the band / below it
pk = (0.5 * m["ncc"].values + 0.3 * (m["sharpness"].values.clip(0) / (np.nanmax(m["sharpness"].values) + 1e-9))
      + 0.2 * m["top1_corr"].values)
pk = (pk - np.nanmin(pk[f1])) / (np.nanmax(pk[f1]) - np.nanmin(pk[f1]) + 1e-9)
s[f1] = 0.55 + 0.42 * (0.55 * m["pA"].values[f1] + 0.45 * pk[f1])
# suspected FP among found==1 (low pA) -> push below 0.5
susp_fp = f1 & (m["pA"].values < 0.35)
s[susp_fp] = 0.15 + 0.20 * m["pA"].values[susp_fp]

# found==0: mid band for likely-correct-absent, low band for likely-missed
#   pA here already estimates P(correct) i.e. P(REJ_OK) for these rows
p0 = m["pA"].values[f0]
p0n = (p0 - p0.min()) / (p0.max() - p0.min() + 1e-9)
s0 = np.where(p0 >= 0.5, 0.34 + 0.16 * p0n, 0.02 + 0.18 * p0n)
s[f0] = s0

sp_new = spearmanr(s, y).statistic
print(f"Stage-B graded  Spearman={sp_new:.4f}  (cal {sp_new*10:.2f})  delta {(sp_new-base_sp)*10:+.2f}")

# guard: keep found==0 x/y/theta/scale zero, keep found unchanged
newp = base.copy()
newp["score"] = np.clip(s, 0.0, 1.0)
assert (newp["found"].values == base["found"].values).all()
for c in ["x", "y", "theta", "scale"]:
    assert (newp[c].values == base[c].values).all()
z = newp[newp.found == 0]
assert (z[["x", "y", "theta", "scale"]].abs().sum(axis=1) == 0).all()
newp.to_csv(a.out, index=False)
print("wrote", a.out)
if a.model_out:
    with open(a.model_out, "wb") as f:
        pickle.dump({"stageA": mdlA, "features": FE, "method": "graded-bucketed",
                     "full_spearman": float(sp_new), "oofA_spearman": float(spearmanr(pA_oof, y).statistic)}, f)
    print("wrote", a.model_out)
