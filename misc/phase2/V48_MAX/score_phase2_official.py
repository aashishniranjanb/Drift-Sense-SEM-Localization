"""OFFICIAL Phase 2 scorer — implements the Applied Materials_Phase 2_Task.pptx
rubric verbatim. This replaces the (incorrect) accepted-only / Spearman scoring
used elsewhere in the repo.

Slide 7 (localization): tier credit 1.00/0.80/0.60/0.40/0.00 at 1/2/3/5 px,
  Euclidean centre error. "Set score = mean credit over that set" -> mean over
  EVERY present pair in the set (rejected present pairs score 0).
  Total = 40 * (0.45*A_mean + 0.55*B_mean).
Slide 7 (pose): credit 1.00/0.60/0.30. Scale |s^-s|/s at 1%/2%/5%. Rotation
  |th^-th| at 0.25/0.5/1.0 deg. "Scored only where localization credit > 0."
  Scale 10 pts, rotation 10 pts.
Slide 8 (rejection): F1 on the found flag across all 180 pairs, positive class =
  found==0 (a team that never rejects scores 0). 15 * F1.
Slide 6 (calibration): AUC of score vs per-pair correctness on the blind set.
  10 * AUC.
Slide "Output Contract": efficiency 5 pts at median <= 5 s/pair; per-pair hard
  timeout 20 s -> that pair scores zero. Docs/generator/citations = 10.
Bonus (cannot lift total above 100 for ranking): +6 if Set D credit >= 0.40 and
  Sets A-C mean credit >= 0.50; +4 if rejection F1 >= 0.90.

Usage:
  python score_phase2_official.py --gt data/phase2_dev/pairs.csv \
      --pred FINAL_SUBMISSION/verification/predictions.csv [--runtime-median S] [--pose-weight ab|flat]
"""
import argparse, json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def loc_credit(e):
    if np.isnan(e):
        return 0.0
    if e <= 1.0:
        return 1.0
    if e <= 2.0:
        return 0.8
    if e <= 3.0:
        return 0.6
    if e <= 5.0:
        return 0.4
    return 0.0


def scale_credit(p):   # p = |s^ - s| / s  (fraction, not %)
    if np.isnan(p):
        return 0.0
    if p <= 0.01:
        return 1.0
    if p <= 0.02:
        return 0.6
    if p <= 0.05:
        return 0.3
    return 0.0


def rot_credit(d):     # d = |th^ - th| in degrees
    if np.isnan(d):
        return 0.0
    if d <= 0.25:
        return 1.0
    if d <= 0.5:
        return 0.6
    if d <= 1.0:
        return 0.3
    return 0.0


def score(gt_csv, pred_csv, runtime_median=None, per_pair_runtime=None, pose_weight="flat", docs=10.0):
    gt = pd.read_csv(gt_csv)
    pr = pd.read_csv(pred_csv)
    m = gt.merge(pr, on="pair_id", suffixes=("", "_p"))
    assert len(m) == len(gt) == 180, f"row mismatch {len(m)}/{len(gt)}"

    m["pf"] = m["found"].astype(int)
    m["gf"] = m["gt_found"].astype(int)
    present = m["gf"] == 1
    m["loc_err"] = np.where(present & (m["pf"] == 1),
                            np.hypot(m["x"] - m["gt_x"], m["y"] - m["gt_y"]), np.nan)
    # rejected present pair -> huge error -> credit 0
    m["loc_err"] = np.where(present & (m["pf"] == 0), 1e9, m["loc_err"])
    m["lc"] = m["loc_err"].apply(loc_credit)

    # ---- localization ----
    A = m.loc[(m.set_type == "SetA") & present, "lc"]
    B = m.loc[(m.set_type == "SetB") & present, "lc"]
    A_mean, B_mean = float(A.mean()), float(B.mean())
    loc_pts = 40.0 * (0.45 * A_mean + 0.55 * B_mean)

    # ---- pose (eligible = present, accepted, loc credit > 0) ----
    elig = present & (m["pf"] == 1) & (m["lc"] > 0)
    sub = m[elig].copy()
    sub["sc"] = (np.abs(sub["scale"] - sub["gt_scale"]) / sub["gt_scale"]).apply(scale_credit)
    sub["rc"] = np.abs(sub["theta"] - sub["gt_theta"]).apply(rot_credit)
    if pose_weight == "ab":
        sA = sub[sub.set_type == "SetA"]; sB = sub[sub.set_type == "SetB"]
        scl = 0.45 * (sA["sc"].mean() if len(sA) else 0) + 0.55 * (sB["sc"].mean() if len(sB) else 0)
        rot = 0.45 * (sA["rc"].mean() if len(sA) else 0) + 0.55 * (sB["rc"].mean() if len(sB) else 0)
    else:
        scl = float(sub["sc"].mean()) if len(sub) else 0.0
        rot = float(sub["rc"].mean()) if len(sub) else 0.0
    pose_pts = 10.0 * scl + 10.0 * rot

    # ---- rejection: F1 on found flag, positive class = found==0, all 180 ----
    yt = (m["gf"] == 0).astype(int).values
    yp = (m["pf"] == 0).astype(int).values
    tp = int(((yt == 1) & (yp == 1)).sum())   # absent correctly rejected
    fp = int(((yt == 0) & (yp == 1)).sum())   # present wrongly rejected
    fn = int(((yt == 1) & (yp == 0)).sum())   # absent wrongly accepted
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    rej_pts = 15.0 * f1

    # ---- calibration: AUC(score, per-pair correctness) ----
    # correct = present localized with credit>0, OR absent correctly rejected
    m["correct"] = (((present) & (m["pf"] == 1) & (m["lc"] > 0)) | ((~present) & (m["pf"] == 0))).astype(int)
    s = m["score"].astype(float).values
    y = m["correct"].values
    auc = float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")
    cal_pts = 10.0 * (auc if not np.isnan(auc) else 0.5)

    # ---- efficiency ----
    if per_pair_runtime is not None:
        rt = np.asarray(per_pair_runtime, float)
        med = float(np.median(rt))
        n_timeout = int((rt > 20.0).sum())
    else:
        med = runtime_median
        n_timeout = 0
    if med is None:
        eff_pts = float("nan")
    else:
        eff_pts = 5.0 if med <= 5.0 else max(0.0, 5.0 - (med - 5.0))

    total = loc_pts + pose_pts + rej_pts + cal_pts + (eff_pts if not np.isnan(eff_pts) else 0.0) + docs

    # ---- bonus eligibility ----
    d = m[m.set_type == "SetD"]
    setD_credit = float(d["lc"].mean()) if len(d) else None
    ac_credit = float(m.loc[m.set_type.isin(["SetA", "SetB", "SetC"]) & present, "lc"].mean())
    bonus6 = (setD_credit is not None and setD_credit >= 0.40 and A_mean >= 0.50 and B_mean >= 0.50)
    bonus4 = f1 >= 0.90

    return {
        "TOTAL_100": round(total, 3),
        "components": {
            "localization_40": round(loc_pts, 3),
            "pose_20": round(pose_pts, 3),
            "rejection_15": round(rej_pts, 3),
            "calibration_10": round(cal_pts, 3),
            "efficiency_5": (round(eff_pts, 3) if not np.isnan(eff_pts) else None),
            "docs_10": docs,
        },
        "localization": {"A_mean_credit": round(A_mean, 4), "B_mean_credit": round(B_mean, 4),
                         "A_present": int(present[m.set_type == "SetA"].sum()),
                         "B_present": int(present[m.set_type == "SetB"].sum()),
                         "A_localized_gt0": int(((m.set_type == "SetA") & elig).sum()),
                         "B_localized_gt0": int(((m.set_type == "SetB") & elig).sum())},
        "pose": {"scale_credit_mean": round(float(scl), 4), "rot_credit_mean": round(float(rot), 4),
                 "eligible_pairs": int(elig.sum()), "weighting": pose_weight},
        "rejection": {"f1": round(f1, 4), "precision": round(prec, 4), "recall": round(rec, 4),
                      "TP_absent_rejected": tp, "FP_present_rejected": fp, "FN_absent_accepted": fn},
        "calibration": {"auc": round(auc, 4) if not np.isnan(auc) else None,
                        "n_correct": int(y.sum()), "n_incorrect": int((1 - y).sum())},
        "efficiency": {"median_s_per_pair": med, "n_timeouts_gt20s": n_timeout},
        "bonus": {"setD_credit": setD_credit, "AC_present_credit": round(ac_credit, 4),
                  "eligible_+6_setD": bool(bonus6), "eligible_+4_rejF1": bool(bonus4)},
    }, m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="data/phase2_dev/pairs.csv")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--runtime-median", type=float, default=None)
    ap.add_argument("--pose-weight", choices=["flat", "ab"], default="flat")
    ap.add_argument("--taxo-out", default=None)
    a = ap.parse_args()
    res, m = score(a.gt, a.pred, runtime_median=a.runtime_median, pose_weight=a.pose_weight)
    print(json.dumps(res, indent=2))
    if a.taxo_out:
        m["loc_credit"] = m["lc"]
        m[["pair_id", "set_type", "gf", "pf", "loc_err", "loc_credit", "theta", "gt_theta",
           "scale", "gt_scale", "score", "correct"]].to_csv(a.taxo_out, index=False)
