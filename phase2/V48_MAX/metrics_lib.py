"""
V48_MAX unified metrics.

Computes, from a ground-truth CSV (data/phase2_dev/pairs.csv format:
pair_id,set_type,reference_path,search_path,gt_x,gt_y,gt_theta,gt_scale,gt_found)
and a predictions CSV (pair_id,x,y,theta,scale,found,score):

  - V22 competition /100 total score (localization 40, pose 20, rejection 15,
    calibration 10, efficiency 5, docs 10) via phase2/V22_CHAMPIONSHIP/scorer.py
  - localization tiers per Set (<=1px, <=2px, <=5px, median)
  - pose MAE (rotation deg, scale % and abs)
  - rejection confusion matrix (positive class = ABSENT / found==0)
  - calibration: ROC AUC (primary), PR AUC, Spearman, Brier, log loss, ECE
  - score-ordering AUCs: correct-vs-wrong (present), present-vs-absent,
    correct-vs-all-incorrect
  - failure taxonomy per pair

Usage:
  python metrics_lib.py --gt data/phase2_dev/pairs.csv \
      --pred phase2/V48_MAX/BASELINE/predictions.csv \
      --out-dir phase2/V48_MAX/BASELINE --runtime-median 7.0 --label BASELINE
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(ROOT, "phase2", "V22_CHAMPIONSHIP"))
from scorer import compute_competition_score  # noqa: E402

from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
)


def _loc_err(row):
    if int(row["found"]) == 1 and int(row["gt_found"]) == 1:
        return float(np.hypot(row["x"] - row["gt_x"], row["y"] - row["gt_y"]))
    return np.nan


def _tier(e):
    if np.isnan(e):
        return "n/a"
    if e <= 1.0:
        return "<=1px"
    if e <= 2.0:
        return "<=2px"
    if e <= 5.0:
        return "<=5px"
    return ">5px"


def _correctness(row):
    # aligned with V22 scorer definition
    if int(row["gt_found"]) == 1 and int(row["found"]) == 1 and not np.isnan(row["loc_err"]) and row["loc_err"] <= 5.0:
        return 1
    if int(row["gt_found"]) == 0 and int(row["found"]) == 0:
        return 1
    return 0


def _failure_mode(row):
    gf, pf = int(row["gt_found"]), int(row["found"])
    if gf == 1 and pf == 1:
        e = row["loc_err"]
        if np.isnan(e):
            return "PRESENCE_FALSE_NEGATIVE"
        if e <= 1.0:
            return "SUBPIXEL_SUCCESS"
        if e <= 5.0:
            return "IN_BOUNDS_SUCCESS"
        return "PERIODIC_REPLICA"
    if gf == 0 and pf == 0:
        return "REJECTION_SUCCESS"
    if gf == 1 and pf == 0:
        return "PRESENCE_FALSE_NEGATIVE"
    return "ABSENCE_FALSE_POSITIVE"


def _safe_auc(y, s):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def _safe_ap(y, s):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, s))


def _ece(y, p, n_bins=10):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        e += (m.sum() / len(p)) * abs(y[m].mean() - p[m].mean())
    return float(e)


def compute(gt_csv, pred_csv, out_dir, runtime_median, label):
    gt = pd.read_csv(gt_csv)
    pred = pd.read_csv(pred_csv)
    m = pd.merge(gt, pred, on="pair_id", how="inner", suffixes=("", "_pred"))
    assert len(m) == len(gt), f"merge lost rows: {len(m)} vs {len(gt)}"

    m["loc_err"] = m.apply(_loc_err, axis=1)
    m["tier"] = m["loc_err"].apply(_tier)
    m["correctness"] = m.apply(_correctness, axis=1)
    m["failure_mode"] = m.apply(_failure_mode, axis=1)

    # ---- V22 competition score ----
    comp_in = m.rename(columns={"found": "pred_found"}).copy()
    comp = compute_competition_score(comp_in, runtime_median=runtime_median)

    # ---- localization tiers per set ----
    loc = {}
    for st in ["SetA", "SetB", "SetC"]:
        sub = m[(m["set_type"] == st) & (m["gt_found"] == 1) & (m["found"] == 1)]
        errs = sub["loc_err"].dropna().values
        present_n = int(((m["set_type"] == st) & (m["gt_found"] == 1)).sum())
        loc[st] = {
            "present_pairs": present_n,
            "localized_pairs": int(len(errs)),
            "le_1px_pct": float(np.mean(errs <= 1.0) * 100) if len(errs) else 0.0,
            "le_2px_pct": float(np.mean(errs <= 2.0) * 100) if len(errs) else 0.0,
            "le_5px_pct": float(np.mean(errs <= 5.0) * 100) if len(errs) else 0.0,
            # coverage-aware: fraction of ALL present pairs in set that land <=5px
            "le_5px_of_present_pct": float(np.sum(errs <= 5.0) / present_n * 100) if present_n else 0.0,
            "median_px": float(np.median(errs)) if len(errs) else 0.0,
        }

    # ---- pose MAE (over loc<=5px present pairs) ----
    pose = {}
    for st in ["SetA", "SetB"]:
        sub = m[(m["set_type"] == st) & (m["gt_found"] == 1) & (m["found"] == 1) & (m["loc_err"] <= 5.0)]
        if len(sub):
            pose[st] = {
                "n": int(len(sub)),
                "rot_mae_deg": float(np.mean(np.abs(sub["theta"] - sub["gt_theta"]))),
                "scale_mae_abs": float(np.mean(np.abs(sub["scale"] - sub["gt_scale"]))),
                "scale_mae_pct": float(np.mean(np.abs(sub["scale"] - sub["gt_scale"]) / sub["gt_scale"] * 100)),
            }
        else:
            pose[st] = {"n": 0, "rot_mae_deg": 0.0, "scale_mae_abs": 0.0, "scale_mae_pct": 0.0}
    allsub = m[(m["gt_found"] == 1) & (m["found"] == 1) & (m["loc_err"] <= 5.0)]
    pose["ALL"] = {
        "n": int(len(allsub)),
        "rot_mae_deg": float(np.mean(np.abs(allsub["theta"] - allsub["gt_theta"]))) if len(allsub) else 0.0,
        "scale_mae_abs": float(np.mean(np.abs(allsub["scale"] - allsub["gt_scale"]))) if len(allsub) else 0.0,
        "scale_mae_pct": float(np.mean(np.abs(allsub["scale"] - allsub["gt_scale"]) / allsub["gt_scale"] * 100)) if len(allsub) else 0.0,
    }

    # ---- rejection confusion (positive class = ABSENT, found==0) ----
    y_abs_true = (m["gt_found"] == 0).astype(int).values
    y_abs_pred = (m["found"] == 0).astype(int).values
    tp = int(np.sum((y_abs_true == 1) & (y_abs_pred == 1)))  # correctly rejected absent
    tn = int(np.sum((y_abs_true == 0) & (y_abs_pred == 0)))  # correctly kept present
    fp = int(np.sum((y_abs_true == 0) & (y_abs_pred == 1)))  # present wrongly rejected (FN of detection)
    fn = int(np.sum((y_abs_true == 1) & (y_abs_pred == 0)))  # absent wrongly accepted (FP of detection)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    rejection = {
        "positive_class": "ABSENT (found==0)",
        "TP_absent_rejected": tp, "TN_present_kept": tn,
        "FP_present_wrongly_rejected": fp, "FN_absent_wrongly_accepted": fn,
        "precision": prec, "recall": rec, "f1": f1,
        # detection view
        "detection_TP": tn, "detection_FP": fn, "detection_FN": fp,
        "present_recall": float(((m["gt_found"] == 1) & (m["found"] == 1)).sum() / (m["gt_found"] == 1).sum()),
    }

    # ---- calibration / ordering ----
    s = m["score"].astype(float).values
    corr = m["correctness"].values
    present = m[m["gt_found"] == 1]
    # correct-vs-wrong among PRESENT & pred_found==1
    pp = present[present["found"] == 1]
    cvw_y = (pp["loc_err"] <= 5.0).astype(int).values
    cvw_s = pp["score"].astype(float).values
    # present-vs-absent over all
    pva_y = (m["gt_found"] == 1).astype(int).values
    # correct-vs-all-incorrect over all
    p_clip = np.clip(s, 1e-6, 1 - 1e-6)
    calibration = {
        "official_roc_auc_score_vs_correctness": _safe_auc(corr, s),
        "pr_auc_score_vs_correctness": _safe_ap(corr, s),
        "spearman_score_vs_correctness": float(spearmanr(s, corr).statistic) if not np.isnan(spearmanr(s, corr).statistic) else 0.0,
        "brier_score_vs_correctness": float(brier_score_loss(corr, p_clip)),
        "log_loss_vs_correctness": float(log_loss(corr, p_clip, labels=[0, 1])),
        "ece_10bin": _ece(corr, p_clip),
        "auc_correct_vs_wrong_present": _safe_auc(cvw_y, cvw_s),
        "auc_present_vs_absent": _safe_auc(pva_y, s),
        "auc_correct_vs_all_incorrect": _safe_auc(corr, s),
        "n_score_ties": int(len(s) - len(np.unique(np.round(s, 6)))),
        "n_distinct_scores": int(len(np.unique(np.round(s, 6)))),
    }

    # ---- taxonomy ----
    tax = m[["pair_id", "set_type", "gt_found", "found", "loc_err", "tier",
             "theta", "gt_theta", "scale", "gt_scale", "score", "correctness", "failure_mode"]].copy()
    tax = tax.rename(columns={"found": "pred_found"})
    tax_path = os.path.join(out_dir, "failure_taxonomy.csv")
    tax.to_csv(tax_path, index=False)
    tax_counts = tax["failure_mode"].value_counts().to_dict()

    metrics = {
        "label": label,
        "gt_csv": gt_csv,
        "pred_csv": pred_csv,
        "n_pairs": int(len(m)),
        "runtime_median_used": runtime_median,
        "competition_v22": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in comp.items()},
        "localization": loc,
        "pose": pose,
        "rejection": rejection,
        "calibration": calibration,
        "failure_taxonomy_counts": tax_counts,
    }

    cm_path = os.path.join(out_dir, "confusion_matrix.json")
    with open(cm_path, "w") as f:
        json.dump({"rejection": rejection,
                   "detection_matrix": {"TP_present_found": tn, "TN_absent_rejected": tp,
                                        "FP_absent_found": fn, "FN_present_missed": fp}}, f, indent=2)

    mpath = os.path.join(out_dir, "metrics.json")
    with open(mpath, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def _print(metrics):
    c = metrics["competition_v22"]
    print("=" * 70)
    print(f" {metrics['label']}  |  {metrics['n_pairs']} pairs")
    print("=" * 70)
    print(f" TOTAL SCORE (V22 /100):     {c['Total Score']:.2f}")
    print(f"   Localization (40):        {c['Localization (40)']:.2f}")
    print(f"   Pose (20):                {c['Pose (20)']:.2f}")
    print(f"   Rejection (15):           {c['Rejection (15)']:.2f}   (F1={c['Rejection F1']:.4f})")
    print(f"   Calibration (10):         {c['Calibration (10)']:.2f}   (AUC={c['Calibration AUC']:.4f})")
    print(f"   Efficiency (5):           {c['Efficiency (5)']:.2f}")
    print(f"   Docs (10):                {c['Docs (10)']:.2f}")
    print("-" * 70)
    for st in ["SetA", "SetB", "SetC"]:
        L = metrics["localization"][st]
        print(f" {st}: present={L['present_pairs']:3d} localized={L['localized_pairs']:3d} "
              f"<=1px={L['le_1px_pct']:5.1f}% <=5px={L['le_5px_pct']:5.1f}% "
              f"(<=5px/present={L['le_5px_of_present_pct']:5.1f}%) med={L['median_px']:.2f}px")
    print("-" * 70)
    P = metrics["pose"]["ALL"]
    print(f" Pose ALL (n={P['n']}): rot MAE={P['rot_mae_deg']:.4f} deg  scale MAE={P['scale_mae_abs']:.4f} ({P['scale_mae_pct']:.2f}%)")
    R = metrics["rejection"]
    print(f" Rejection: TP(abs rej)={R['TP_absent_rejected']} TN(pres kept)={R['TN_present_kept']} "
          f"FP(pres rej)={R['FP_present_wrongly_rejected']} FN(abs acc)={R['FN_absent_wrongly_accepted']} "
          f"F1={R['f1']:.4f} present_recall={R['present_recall']:.4f}")
    K = metrics["calibration"]
    print(f" Calibration: ROC AUC={K['official_roc_auc_score_vs_correctness']:.4f} PR AUC={K['pr_auc_score_vs_correctness']:.4f} "
          f"Spearman={K['spearman_score_vs_correctness']:.4f} Brier={K['brier_score_vs_correctness']:.4f} "
          f"logloss={K['log_loss_vs_correctness']:.4f} ECE={K['ece_10bin']:.4f}")
    print(f" Ordering AUC: corr-vs-wrong(present)={K['auc_correct_vs_wrong_present']:.4f} "
          f"present-vs-absent={K['auc_present_vs_absent']:.4f} corr-vs-all-incorrect={K['auc_correct_vs_all_incorrect']:.4f}")
    print(f" Score spread: {K['n_distinct_scores']} distinct / {metrics['n_pairs']}  ({K['n_score_ties']} collisions)")
    print(" Failure taxonomy:", metrics["failure_taxonomy_counts"])
    print("=" * 70)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="data/phase2_dev/pairs.csv")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--runtime-median", type=float, default=7.0)
    ap.add_argument("--label", default="RUN")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    _print(compute(a.gt, a.pred, a.out_dir, a.runtime_median, a.label))
