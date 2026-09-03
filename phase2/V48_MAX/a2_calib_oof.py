"""V48 analysis 2: OOF calibration ceiling.

Score column only. found / x / y / theta / scale frozen from V41 FINAL.
Target = benchmark correctness label. Metric = Spearman(score, correctness) on all 180.
5-fold stratified OOF (strata: absent / A_present / B_present). No full-data fit reported as result.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression
import warnings; warnings.filterwarnings("ignore")

pairs = pd.read_csv("data/phase2_dev/pairs.csv")
v41 = pd.read_csv("phase2/V41_CALIBRATION/FINAL/v41_predictions.csv")
rej = pd.read_csv("phase2/V27_REJECTION/v25_rejection_features.csv")
v47 = pd.read_csv("phase2/V47_RESEARCH/v47_candidate_cache/features.csv")

m = pairs.merge(v41, on="pair_id")
m = m.merge(rej[["pair_id", "top1_score", "margin", "top1_corr", "top1_ctx", "top1_neigh", "top1_grad", "mode_strong"]], on="pair_id", how="left")
V47C = ["ncc", "grad", "ctx", "phase", "ncc_pct", "grad_pct", "prom10_ncc", "z10_ncc",
        "comp10", "comp20", "d1", "d2", "dist_center", "dist_border", "sharpness", "delta_ncc", "dist_v25_v46"]
m = m.merge(v47[["pair_id"] + [c for c in V47C if c in v47.columns]], on="pair_id", how="left")
m = m.fillna(0.0)

m["loc_err"] = np.where((m.found == 1) & (m.gt_found == 1), np.hypot(m.x - m.gt_x, m.y - m.gt_y), np.nan)
m["correct"] = (((m.gt_found == 1) & (m.found == 1) & (m.loc_err <= 5.0)) |
                ((m.gt_found == 0) & (m.found == 0))).astype(int)
m["strat"] = np.where(m.gt_found == 0, "absent", np.where(m.set_type == "SetA", "Apres", "Bpres"))

base_sp = spearmanr(m["score"], m["correct"]).statistic
print(f"n=180  correct={m.correct.sum()}  baseline Spearman={base_sp:.4f}  (V41 FINAL score)")
print(f"oracle Spearman (SUBPIX/INB=1, REJ=.3, FN=.1, FP=0) reference = 0.8947\n")

FEATS = ["score", "found", "top1_score", "margin", "top1_corr", "top1_ctx", "top1_neigh",
         "top1_grad", "mode_strong"] + [c for c in V47C if c in m.columns]
X = m[FEATS].values.astype(float)
y = m["correct"].values
strat = m["strat"].values

def oof(make_model, scale=False):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    p = np.zeros(len(m))
    for tr, te in skf.split(X, strat):
        Xtr, Xte = X[tr], X[te]
        if scale:
            sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        mdl = make_model(); mdl.fit(Xtr, y[tr])
        p[te] = mdl.predict_proba(Xte)[:, 1]
    return p

configs = {
    "LR C=0.05": (lambda: LogisticRegression(C=0.05, max_iter=5000), True),
    "LR C=0.1": (lambda: LogisticRegression(C=0.1, max_iter=5000), True),
    "LR C=0.3": (lambda: LogisticRegression(C=0.3, max_iter=5000), True),
    "HGB d2 lr.03": (lambda: HistGradientBoostingClassifier(max_depth=2, learning_rate=0.03, max_iter=300, min_samples_leaf=20, l2_regularization=1.0, random_state=42), False),
    "HGB d3 lr.05": (lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=20, l2_regularization=1.0, random_state=42), False),
}
results = {}
for name, (mk, sc) in configs.items():
    p = oof(mk, sc)
    # keep found frozen; found=0 must keep x/y/theta/scale=0 but SCORE can be nonzero
    sp_raw = spearmanr(p, y).statistic
    # blend with V41 score (monotone-ish safety) then re-check
    for w in (1.0, 0.7, 0.5):
        pb = w * p + (1 - w) * m["score"].values
        sp_b = spearmanr(pb, y).statistic
        results[f"{name} w={w}"] = sp_b
    print(f"{name:16s} OOF Spearman={sp_raw:.4f}  (blend .7={results[name+' w=0.7']:.4f}, .5={results[name+' w=0.5']:.4f})")

best = max(results, key=results.get)
print(f"\nBEST: {best} -> Spearman={results[best]:.4f}  => calibration pts {results[best]*10:.2f} (was {base_sp*10:.2f})")

# also: pure found=0 reorder only (found=1 score untouched) with OOF HGB restricted to f0 rows
print("\n--- found=0-only reorder (found=1 score = V41 untouched) ---")
f0 = m[m.found == 0].reset_index(drop=True)
Xf0 = f0[[c for c in FEATS if c != "found"]].values.astype(float)
yf0 = f0["correct"].values  # 1 = REJ_OK, 0 = FN
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
pf0 = np.zeros(len(f0))
strat_f0 = np.where(f0.set_type == "SetA", "A", np.where(f0.set_type == "SetB", "B", "C"))
try:
    for tr, te in skf.split(Xf0, yf0):
        mdl = HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05, max_iter=200, min_samples_leaf=10, random_state=0)
        mdl.fit(Xf0[tr], yf0[tr]); pf0[te] = mdl.predict_proba(Xf0[te])[:, 1]
    auc_f0 = spearmanr(pf0, yf0).statistic
    for band in (0.2, 0.3, 0.4):
        s = m["score"].values.astype(float).copy()
        s[(m.found == 0).values] = band * pf0
        sp = spearmanr(s, y).statistic
        print(f"  band={band}: within-f0 spearman={auc_f0:.3f}  ALL-180 Spearman={sp:.4f} (delta {sp-base_sp:+.4f})")
except Exception as e:
    print("  f0 CV failed:", e)
