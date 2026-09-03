import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

v39 = pd.read_csv("phase2/V39_POSE/v39_predictions.csv")
pairs = pd.read_csv("data/phase2_dev/pairs.csv")
rej27 = pd.read_csv("phase2/V27_REJECTION/v25_rejection_features.csv")
v39r = pd.read_csv("phase2/V39_POSE/v39_results.csv")

merged = pd.merge(pairs, v39, on="pair_id")
merged = pd.merge(merged, rej27[["pair_id","top1_score","margin","top1_corr","top1_ctx","top1_neigh","top1_grad","mode_strong"]], on="pair_id", how="left")
merged = pd.merge(merged, v39r[["pair_id","displacement","fallback","elapsed_ms"]], on="pair_id", how="left")
merged["loc_err"] = np.hypot(merged["gt_x"] - merged["x"], merged["gt_y"] - merged["y"])
merged["correct_present"] = ((merged["found"]==1) & (merged["gt_found"]==1) & (merged["loc_err"]<=5.0)).astype(int)
merged["correct_absent"] = ((merged["found"]==0) & (merged["gt_found"]==0)).astype(int)
merged["correct"] = np.maximum(merged["correct_present"], merged["correct_absent"])

# Strat label: A_present, B_present, absent, wrong_found (treat as present)
def get_strat(row):
    if row["gt_found"] == 0:
        return "absent"
    elif row["set_type"] == "SetA":
        return "A_present"
    else:
        return "B_present"
merged["strat_label"] = merged.apply(get_strat, axis=1)

print("=== BASELINE: V39 frozen score ===")
auc_base = roc_auc_score(merged["correct"], merged["score"])
sp_base = spearmanr(merged["correct"], merged["score"])[0]
print(f"AUC={auc_base:.4f} Spearman={sp_base:.4f}")
print()

# Feature set for evidence-aware calibration
feature_cols = ["score", "top1_score", "margin", "top1_corr", "top1_ctx", "top1_neigh", "top1_grad", "mode_strong"]
merged["found_flag"] = merged["found"].astype(float)
merged["score_x_margin"] = merged["score"] * merged["margin"]
merged["score_x_top1"] = merged["score"] * merged["top1_score"]
merged["log_score"] = np.log(merged["score"].clip(1e-6, 1 - 1e-6))
extended_features = feature_cols + ["found_flag", "score_x_margin", "score_x_top1", "log_score"]
merged[extended_features] = merged[extended_features].fillna(0)

print("=== OOF CALIBRATION EXPERIMENTS (5-fold Stratified) ===")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y = merged["correct"].values
X_full = merged[extended_features].values
strat = merged["strat_label"].values

results = {}

# V41-A: Platt (Logistic on raw score only)
oof_platt = np.zeros(len(y))
for tr, va in skf.split(X_full, strat):
    lr = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs")
    lr.fit(X_full[tr, 0:1], y[tr])  # only raw score
    oof_platt[va] = lr.predict_proba(X_full[va, 0:1])[:, 1]
auc_platt = roc_auc_score(y, oof_platt)
sp_platt = spearmanr(y, oof_platt)[0]
results["Platt (score only)"] = {"AUC": auc_platt, "Spearman": sp_platt}
print(f"V41-A Platt: AUC={auc_platt:.4f} Spearman={sp_platt:.4f}")

# V41-B: Isotonic on raw score
oof_iso = np.zeros(len(y))
for tr, va in skf.split(X_full, strat):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(X_full[tr, 0], y[tr])
    oof_iso[va] = iso.predict(X_full[va, 0])
oof_iso = np.clip(oof_iso, 0, 1)
auc_iso = roc_auc_score(y, oof_iso)
sp_iso = spearmanr(y, oof_iso)[0]
results["Isotonic"] = {"AUC": auc_iso, "Spearman": sp_iso}
print(f"V41-B Isotonic: AUC={auc_iso:.4f} Spearman={sp_iso:.4f}")

# V41-C: Temperature scaling on logit of score
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

best_T = 1.0
best_T_auc = 0
oof_temp_all = {}
for T in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]:
    eps = 1e-6
    s = np.clip(merged["score"].values, eps, 1 - eps)
    logit = np.log(s / (1 - s))
    cal_score = sigmoid(logit / T)
    auc_t = roc_auc_score(y, cal_score)
    sp_t = spearmanr(y, cal_score)[0]
    oof_temp_all[T] = {"AUC": auc_t, "Spearman": sp_t, "scores": cal_score}
    if auc_t > best_T_auc:
        best_T_auc = auc_t
        best_T = T
print(f"V41-C Temperature best T={best_T}: AUC={best_T_auc:.4f}")

# V41-D: Logistic with evidence features
oof_logit_ev = np.zeros(len(y))
for C in [0.1, 0.3, 1.0]:
    oof_tmp = np.zeros(len(y))
    sc = StandardScaler()
    for tr, va in skf.split(X_full, strat):
        Xtr = sc.fit_transform(X_full[tr])
        Xva = sc.transform(X_full[va])
        lr = LogisticRegression(C=C, max_iter=5000, solver="lbfgs")
        lr.fit(Xtr, y[tr])
        oof_tmp[va] = lr.predict_proba(Xva)[:, 1]
    auc_ev = roc_auc_score(y, oof_tmp)
    sp_ev = spearmanr(y, oof_tmp)[0]
    print(f"V41-D Logistic evidence (C={C}): AUC={auc_ev:.4f} Spearman={sp_ev:.4f}")
    if auc_ev > roc_auc_score(y, oof_logit_ev) or oof_logit_ev.sum() == 0:
        oof_logit_ev = oof_tmp
        results["Logistic Evidence"] = {"AUC": auc_ev, "Spearman": sp_ev}

# V41-E: HGB
from sklearn.ensemble import HistGradientBoostingClassifier
oof_hgb = np.zeros(len(y))
for tr, va in skf.split(X_full, strat):
    hgb = HistGradientBoostingClassifier(max_iter=100, max_depth=2, learning_rate=0.05, l2_regularization=1.0, random_state=42)
    hgb.fit(X_full[tr], y[tr])
    oof_hgb[va] = hgb.predict_proba(X_full[va])[:, 1]
auc_hgb = roc_auc_score(y, oof_hgb)
sp_hgb = spearmanr(y, oof_hgb)[0]
results["HGB depth=2"] = {"AUC": auc_hgb, "Spearman": sp_hgb}
print(f"V41-E HGB: AUC={auc_hgb:.4f} Spearman={sp_hgb:.4f}")

# V41-F: Residual correction (weighted mix)
best_combo = None
best_combo_auc = 0
for alpha in np.arange(0.0, 0.35, 0.05):
    combo = (1 - alpha) * merged["score"].values + alpha * merged["top1_score"].values
    auc_c = roc_auc_score(y, combo)
    sp_c = spearmanr(y, combo)[0]
    spear_vs_raw = spearmanr(combo, merged["score"].values)[0]
    if spear_vs_raw >= 0.90 and auc_c > best_combo_auc:
        best_combo_auc = auc_c
        best_combo = {"alpha": alpha, "AUC": auc_c, "Spearman": sp_c, "spear_vs_raw": spear_vs_raw}
if best_combo:
    print(f"V41-F Best residual mix: alpha={best_combo['alpha']:.2f} AUC={best_combo['AUC']:.4f} Spearman={best_combo['Spearman']:.4f}")
    results["Residual Mix"] = {"AUC": best_combo["AUC"], "Spearman": best_combo["Spearman"]}

print()
print("=== SUMMARY vs BASELINE (AUC=%.4f Spearman=%.4f) ===" % (auc_base, sp_base))
for name, r in sorted(results.items(), key=lambda x: -x[1]["AUC"]):
    delta_auc = r["AUC"] - auc_base
    delta_sp = r["Spearman"] - sp_base
    print(f"  {name:30s}: AUC={r['AUC']:.4f} ({delta_auc:+.4f}) | Spearman={r['Spearman']:.4f} ({delta_sp:+.4f})")
