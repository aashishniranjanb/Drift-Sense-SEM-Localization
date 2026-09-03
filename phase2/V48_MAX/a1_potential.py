"""V48 analysis 1: rescue ceiling + calibration reorder direction check."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import warnings; warnings.filterwarnings("ignore")

pairs = pd.read_csv("data/phase2_dev/pairs.csv")
v41 = pd.read_csv("phase2/V41_CALIBRATION/FINAL/v41_predictions.csv")
rej = pd.read_csv("phase2/V27_REJECTION/v25_rejection_features.csv")
v47 = pd.read_csv("phase2/V47_RESEARCH/v47_candidate_cache/features.csv")

m = pairs.merge(v41, on="pair_id", suffixes=("", "_p"))
m = m.merge(rej[["pair_id", "top1_score", "margin", "top1_corr", "top1_ctx", "top1_neigh", "top1_grad", "mode_strong"]], on="pair_id", how="left")
m = m.merge(v47, on="pair_id", how="left", suffixes=("", "_v47"))
m["loc_err"] = np.where((m.found == 1) & (m.gt_found == 1), np.hypot(m.x - m.gt_x, m.y - m.gt_y), np.nan)

def fm(r):
    if r.gt_found == 1 and r.found == 1:
        return "SUBPIXEL" if r.loc_err <= 1 else ("INBOUND" if r.loc_err <= 5 else "PERIODIC")
    if r.gt_found == 0 and r.found == 0: return "REJ_OK"
    if r.gt_found == 1 and r.found == 0: return "FN"
    return "FP"
m["mode"] = m.apply(fm, axis=1)
print(m["mode"].value_counts().to_dict())

fn = m[m["mode"] == "FN"].copy()
print(f"\n=== {len(fn)} FN pairs — candidate proximity to GT ===")
# v46 pool candidate location vs GT
fn["v46_err"] = np.hypot(fn.v46_cx - fn.gt_x, fn.v46_cy - fn.gt_y)
fn["v25_err"] = np.hypot(fn.v25_cx - fn.gt_x, fn.v25_cy - fn.gt_y)
for col, lab in [("v46_err", "V46 pool top"), ("v25_err", "V25 top")]:
    for thr in (1, 2, 5, 10):
        print(f"  {lab:14s} <= {thr:2d}px: {(fn[col] <= thr).sum():2d} / {len(fn)}")
print(f"\n  FN with SOME candidate (v46 or v25) <=5px: {((fn.v46_err <= 5) | (fn.v25_err <= 5)).sum()}")
print(f"  FN by set: {fn.set_type.value_counts().to_dict()}")
rescuable = fn[(fn.v46_err <= 5) | (fn.v25_err <= 5)]
print(f"  rescuable by set: {rescuable.set_type.value_counts().to_dict()}")

print("\n=== evidence: rescuable FN vs non-rescuable FN (v46 pool) ===")
fn["resc"] = (fn.v46_err <= 5).astype(int)
for c in ["v46_score", "v46_consensus", "ncc", "grad", "ctx", "phase", "ncc_pct", "prom10_ncc",
          "z10_ncc", "comp10", "d1", "dist_center", "dist_border", "sharpness", "delta_ncc",
          "top1_score", "top1_corr", "margin"]:
    if c in fn.columns:
        a = fn[fn.resc == 1][c].median(); b = fn[fn.resc == 0][c].median()
        print(f"  {c:16s} resc={a:8.3f}  nonresc={b:8.3f}")

print("\n=== calibration reorder direction (found=0 group) ===")
f0 = m[m.found == 0].copy()
f0["is_correct"] = (f0["mode"] == "REJ_OK").astype(int)
base_sp = spearmanr(m["score"], (m["mode"].isin(["SUBPIXEL", "INBOUND", "REJ_OK"])).astype(int)).statistic
print(f"baseline Spearman(all 180): {base_sp:.4f}")
for c in ["top1_score", "top1_corr", "top1_ctx", "top1_neigh", "top1_grad", "margin"]:
    rej_med = f0[f0.is_correct == 1][c].median(); fn_med = f0[f0.is_correct == 0][c].median()
    sp_in0 = spearmanr(f0[c], f0["is_correct"]).statistic
    print(f"  {c:12s} REJ_OK_med={rej_med:.3f} FN_med={fn_med:.3f}  spearman_within_f0={sp_in0:+.3f}")

# test correct-direction reorder: score_f0 = k * sigmoid-ish of evidence that ranks REJ_OK high
y_all = (m["mode"].isin(["SUBPIXEL", "INBOUND", "REJ_OK"])).astype(int).values
for k in (0.2, 0.3, 0.4):
    for feat in ("top1_score", "top1_corr"):
        s = m["score"].copy().values.astype(float)
        mask0 = (m.found == 0).values
        v = m.loc[m.found == 0, feat].fillna(0).clip(0, 1).values
        s[mask0] = k * v
        sp = spearmanr(s, y_all).statistic
        print(f"  k={k} feat={feat:11s} -> Spearman={sp:.4f} (delta {sp-base_sp:+.4f})")
