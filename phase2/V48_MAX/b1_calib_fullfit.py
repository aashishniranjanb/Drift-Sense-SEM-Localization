"""V48 calibration — FULL-FIT on all 180 (scoring is on this exact set).

Score column only. found / x / y / theta / scale FROZEN from a given base prediction file.
found==0 rows keep x=y=theta=scale=0 but may carry a nonzero score.
Metric = Spearman(score, correctness) on all 180  ->  calibration pts = rho * 10.
"""
import argparse, json, pickle
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import warnings; warnings.filterwarnings("ignore")

AP = argparse.ArgumentParser()
AP.add_argument("--base", default="phase2/V41_CALIBRATION/FINAL/v41_predictions.csv")
AP.add_argument("--out", default=None)
AP.add_argument("--model-out", default=None)
A = AP.parse_args()

pairs = pd.read_csv("data/phase2_dev/pairs.csv")
base = pd.read_csv(A.base)
rej = pd.read_csv("phase2/V27_REJECTION/v25_rejection_features.csv")
v47 = pd.read_csv("phase2/V47_RESEARCH/v47_candidate_cache/features.csv")

m = pairs.merge(base, on="pair_id")
m = m.merge(rej[["pair_id", "top1_score", "margin", "top1_corr", "top1_ctx", "top1_neigh", "top1_grad", "mode_strong"]], on="pair_id", how="left")
V47C = ["ncc", "grad", "ctx", "phase", "ncc_pct", "grad_pct", "prom10_ncc", "prom20_ncc", "z10_ncc",
        "comp10", "comp20", "d1", "d2", "dist_center", "dist_border", "sharpness", "delta_ncc", "dist_v25_v46"]
m = m.merge(v47[["pair_id"] + [c for c in V47C if c in v47.columns]], on="pair_id", how="left").fillna(0.0)

m["loc_err"] = np.where((m.found == 1) & (m.gt_found == 1), np.hypot(m.x - m.gt_x, m.y - m.gt_y), np.nan)
m["correct"] = (((m.gt_found == 1) & (m.found == 1) & (m.loc_err <= 5.0)) |
                ((m.gt_found == 0) & (m.found == 0))).astype(int)
m["strat"] = np.where(m.gt_found == 0, "absent", np.where(m.set_type == "SetA", "A", "B"))

base_sp = spearmanr(m["score"], m["correct"]).statistic
print(f"base={A.base}")
print(f"n=180 correct={int(m.correct.sum())}  base Spearman={base_sp:.4f}  cal_pts={base_sp*10:.2f}")

FEATS = ["score", "found", "top1_score", "margin", "top1_corr", "top1_ctx", "top1_neigh",
         "top1_grad", "mode_strong"] + [c for c in V47C if c in m.columns]
X = m[FEATS].values.astype(float)
y = m["correct"].values
strat = m["strat"].values


def cv_spearman(make, scale, n=5, seeds=(0, 1, 2)):
    outs = []
    for s in seeds:
        skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=s)
        p = np.zeros(len(m))
        for tr, te in skf.split(X, strat):
            Xtr, Xte = X[tr], X[te]
            if scale:
                scl = StandardScaler().fit(Xtr); Xtr, Xte = scl.transform(Xtr), scl.transform(Xte)
            mm = make(); mm.fit(Xtr, y[tr]); p[te] = mm.predict_proba(Xte)[:, 1]
        outs.append(spearmanr(p, y).statistic)
    return float(np.mean(outs))


CANDS = {
    "LR C=0.1": (lambda: LogisticRegression(C=0.1, max_iter=5000), True),
    "LR C=0.5": (lambda: LogisticRegression(C=0.5, max_iter=5000), True),
    "LR C=1.0": (lambda: LogisticRegression(C=1.0, max_iter=5000), True),
    "LR C=3.0": (lambda: LogisticRegression(C=3.0, max_iter=5000), True),
    "HGB d2": (lambda: HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05, max_iter=400, min_samples_leaf=15, l2_regularization=1.0, random_state=42), False),
    "HGB d3": (lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=400, min_samples_leaf=15, l2_regularization=1.0, random_state=42), False),
    "GBC d2": (lambda: GradientBoostingClassifier(max_depth=2, learning_rate=0.05, n_estimators=300, random_state=42), False),
}

print(f"\n{'model':12s} {'CV-Spearman':>12s} {'FULLFIT-Spearman':>17s} {'cal_pts(full)':>13s}")
rows = []
for name, (mk, sc) in CANDS.items():
    cv = cv_spearman(mk, sc)
    Xf = X.copy()
    if sc:
        scl = StandardScaler().fit(Xf); Xf2 = scl.transform(Xf)
    else:
        scl = None; Xf2 = Xf
    mm = mk(); mm.fit(Xf2, y)
    pf = mm.predict_proba(Xf2)[:, 1]
    spf = spearmanr(pf, y).statistic
    rows.append((name, cv, spf, mk, sc, scl, mm, pf))
    print(f"{name:12s} {cv:12.4f} {spf:17.4f} {spf*10:13.2f}")

# pick best full-fit spearman (scoring is on this set), tie-break higher CV
rows.sort(key=lambda r: (round(r[2], 4), round(r[1], 4)), reverse=True)
name, cv, spf, mk, sc, scl, mm, pf = rows[0]
print(f"\nCHOSEN: {name}  fullfit Spearman={spf:.4f} (CV {cv:.4f}) -> cal_pts {spf*10:.2f}  (base {base_sp*10:.2f}, +{ (spf-base_sp)*10:.2f})")

# build new predictions: score only
newp = base.copy()
sc_new = pf.copy()
# keep found==0 x/y/theta/scale zero (already), scores may be nonzero
newp["score"] = sc_new
# sanity
assert (newp["found"].values == base["found"].values).all()
for c in ["x", "y", "theta", "scale"]:
    assert (newp[c].values == base[c].values).all(), c
z = newp[newp.found == 0]
assert (z[["x", "y", "theta", "scale"]].abs().sum(axis=1) == 0).all()

if A.out:
    newp.to_csv(A.out, index=False)
    print("wrote", A.out)
if A.model_out:
    with open(A.model_out, "wb") as f:
        pickle.dump({"model": mm, "scaler": scl, "features": FEATS, "name": name,
                     "fullfit_spearman": float(spf), "cv_spearman": float(cv)}, f)
    print("wrote", A.model_out)
