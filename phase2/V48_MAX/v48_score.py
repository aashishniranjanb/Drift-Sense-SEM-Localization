"""
V48 official competition scorer — implements phase2/V27_FINAL/SCORER_AUDIT.md
(source of truth: phase2/benchmark_phase2.py methodology).

  Localization 40 : (0.45*SetA_le5% + 0.55*SetB_le5%) * 0.40   [accepted present pairs only]
  Pose         20 : tiered scale+rotation credit over accepted present pairs (loc_err<=5)
  Rejection    15 : RejF1 * 15   (positive class = found==0)
  Calibration  10 : Spearman(score, correctness) * 10
  Efficiency    5 : 5.0 if median per-pair runtime <= 5s
  Compliance   10 : schema + zero-coordinate rule for found==0

Pose credit tiers (from phase2/V22_CHAMPIONSHIP/TASK.md):
  scale %err: <=1 ->1.0, <=2 ->0.75, <=5 ->0.5, else 0
  rot  |err|: <=0.25 ->1.0, <=0.5 ->0.75, <=1.0 ->0.5, else 0
  pose = 10*(0.45*scaleA+0.55*scaleB) + 10*(0.45*rotA+0.55*rotB)

NOTE: benchmark_phase2.py itself only prints Spearman/F1/loc%; the /100 rollup
follows run_v39_benchmark.parse_metrics (loc% * .40, F1 * 15, rho * 10, pose fixed
19.2 in that script). Here pose is computed, not fixed, so small differences from
the report's hard-coded 19.20 are expected and reported explicitly.
"""
import argparse, json, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def loc_tier_pct(errs, thr):
    return float(np.mean(errs <= thr) * 100) if len(errs) else 0.0


def score(gt_csv, pred_csv, runtime_median=None, pose_fixed=None):
    gt = pd.read_csv(gt_csv)
    pr = pd.read_csv(pred_csv)
    m = pd.merge(gt, pr, on="pair_id", suffixes=("", "_p"))
    assert len(m) == len(gt), f"row mismatch {len(m)} vs {len(gt)}"

    m["pf"] = m["found"].astype(int)
    m["gf"] = m["gt_found"].astype(int)
    acc_pres = (m["gf"] == 1) & (m["pf"] == 1)
    m["loc_err"] = np.where(acc_pres, np.hypot(m["x"] - m["gt_x"], m["y"] - m["gt_y"]), np.nan)

    # ---- localization ----
    out = {}
    le5 = {}
    for st in ["SetA", "SetB"]:
        e = m.loc[(m["set_type"] == st) & acc_pres, "loc_err"].dropna().values
        le5[st] = loc_tier_pct(e, 5.0)
        out[f"{st}_le1_pct"] = loc_tier_pct(e, 1.0)
        out[f"{st}_le5_pct"] = le5[st]
        out[f"{st}_median_px"] = float(np.median(e)) if len(e) else 0.0
        out[f"{st}_localized_n"] = int(len(e))
    weighted_loc = 0.45 * le5["SetA"] + 0.55 * le5["SetB"]
    loc_points = weighted_loc * 0.40

    # ---- pose ----
    def sc_credit(p):
        return np.where(p <= 1, 1.0, np.where(p <= 2, 0.75, np.where(p <= 5, 0.5, 0.0)))

    def rot_credit(p):
        return np.where(p <= 0.25, 1.0, np.where(p <= 0.5, 0.75, np.where(p <= 1.0, 0.5, 0.0)))

    pose_parts = {}
    for st in ["SetA", "SetB"]:
        sub = m[(m["set_type"] == st) & acc_pres & (m["loc_err"] <= 5.0)]
        if len(sub):
            s_pct = np.abs(sub["scale"] - sub["gt_scale"]) / sub["gt_scale"] * 100
            r_abs = np.abs(sub["theta"] - sub["gt_theta"])
            pose_parts[st] = (float(np.mean(sc_credit(s_pct.values))), float(np.mean(rot_credit(r_abs.values))))
        else:
            pose_parts[st] = (0.0, 0.0)
    scale_pts = 10.0 * (0.45 * pose_parts["SetA"][0] + 0.55 * pose_parts["SetB"][0])
    rot_pts = 10.0 * (0.45 * pose_parts["SetA"][1] + 0.55 * pose_parts["SetB"][1])
    pose_points_computed = scale_pts + rot_pts
    pose_points = pose_fixed if pose_fixed is not None else pose_points_computed

    # ---- rejection (positive class = found==0) ----
    tp = int(((m["gf"] == 0) & (m["pf"] == 0)).sum())
    fp = int(((m["gf"] == 1) & (m["pf"] == 0)).sum())
    fn = int(((m["gf"] == 0) & (m["pf"] == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    rej_f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    rej_points = rej_f1 * 15.0

    # ---- calibration: Spearman(score, correctness) ----
    def correctness(r):
        if r["gf"] == 1 and r["pf"] == 1 and not np.isnan(r["loc_err"]) and r["loc_err"] <= 5.0:
            return 1
        if r["gf"] == 0 and r["pf"] == 0:
            return 1
        return 0

    m["correct"] = m.apply(correctness, axis=1)
    rho = spearmanr(m["score"].values, m["correct"].values).statistic
    rho = 0.0 if np.isnan(rho) else float(rho)
    cal_points = rho * 10.0

    # ---- efficiency + compliance ----
    eff_points = 5.0 if (runtime_median is None or runtime_median <= 5.0) else max(0.0, 5.0 - (runtime_median - 5.0))
    schema_ok = list(pr.columns[:7]) == ["pair_id", "x", "y", "theta", "scale", "found", "score"]
    z = pr[pr["found"] == 0]
    zero_ok = bool(((z[["x", "y", "theta", "scale"]].abs().sum(axis=1)) == 0).all())
    compliance_points = 10.0 if (schema_ok and zero_ok) else (5.0 if zero_ok else 0.0)

    total = loc_points + pose_points + rej_points + cal_points + eff_points + compliance_points

    # taxonomy
    def fm(r):
        if r["gf"] == 1 and r["pf"] == 1:
            e = r["loc_err"]
            return "SUBPIXEL_SUCCESS" if e <= 1 else ("IN_BOUNDS_SUCCESS" if e <= 5 else "PERIODIC_REPLICA")
        if r["gf"] == 0 and r["pf"] == 0:
            return "REJECTION_SUCCESS"
        if r["gf"] == 1 and r["pf"] == 0:
            return "PRESENCE_FALSE_NEGATIVE"
        return "ABSENCE_FALSE_POSITIVE"

    m["failure_mode"] = m.apply(fm, axis=1)

    res = {
        "pred_csv": pred_csv,
        "TOTAL": round(total, 3),
        "points": {
            "localization_40": round(loc_points, 3),
            "pose_20": round(pose_points, 3),
            "pose_20_computed": round(pose_points_computed, 3),
            "rejection_15": round(rej_points, 3),
            "calibration_10": round(cal_points, 3),
            "efficiency_5": round(eff_points, 3),
            "compliance_10": round(compliance_points, 3),
        },
        "localization": {**out, "weighted_loc_pct": round(weighted_loc, 3)},
        "pose": {"scale_pts": round(scale_pts, 3), "rot_pts": round(rot_pts, 3),
                 "SetA_scale_cr": pose_parts["SetA"][0], "SetA_rot_cr": pose_parts["SetA"][1],
                 "SetB_scale_cr": pose_parts["SetB"][0], "SetB_rot_cr": pose_parts["SetB"][1]},
        "rejection": {"f1": round(rej_f1, 4), "precision": round(prec, 4), "recall": round(rec, 4),
                      "TP_absent_rejected": tp, "FP_present_rejected": fp, "FN_absent_accepted": fn},
        "calibration": {"spearman": round(rho, 4)},
        "runtime_median": runtime_median,
        "taxonomy": m["failure_mode"].value_counts().to_dict(),
    }
    return res, m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="data/phase2_dev/pairs.csv")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--runtime-median", type=float, default=None)
    ap.add_argument("--pose-fixed", type=float, default=None,
                    help="override computed pose points (e.g. 19.20 to match V39 report rollup)")
    ap.add_argument("--taxo-out", default=None)
    a = ap.parse_args()
    res, m = score(a.gt, a.pred, a.runtime_median, a.pose_fixed)
    print(json.dumps(res, indent=2))
    if a.taxo_out:
        m[["pair_id", "set_type", "gf", "pf", "loc_err", "theta", "gt_theta", "scale",
           "gt_scale", "score", "correct", "failure_mode"]].to_csv(a.taxo_out, index=False)
