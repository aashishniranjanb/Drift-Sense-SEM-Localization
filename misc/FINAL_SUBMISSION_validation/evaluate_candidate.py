"""
PERMANENT 180-CASE EVALUATION HARNESS
======================================
Evaluates any predictions CSV or candidate pool against the official 180-pair benchmark.
Computes competition scores, sub-pixel error breakdown, candidate retrieval recall (Top-K),
and strict safety metrics (broken baseline successes, absent false accepts).
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

# Constant competition weights & golden baseline scores
GOLDEN_BASELINE_SCORE = 91.040
GOLDEN_LOC_SCORE = 40.000
GOLDEN_POSE_SCORE = 19.743
GOLDEN_REJ_SCORE = 8.028
GOLDEN_CALIB_SCORE = 8.269
GOLDEN_EFF_SCORE = 5.000
GOLDEN_DOC_SCORE = 10.000


def evaluate_predictions(pred_df, pairs_df, golden_pred_df=None, candidate_pool_df=None):
    merged = pd.merge(pairs_df, pred_df, on="pair_id", suffixes=("_gt", "_pred"))

    if golden_pred_df is not None:
        merged = pd.merge(merged, golden_pred_df, on="pair_id", suffixes=("", "_golden"))

    # 1. Localization Metrics (Set A & Set B)
    set_a = merged[(merged["set_type"] == "SetA") & (merged["gt_found"] == 1)].copy()
    set_b = merged[(merged["set_type"] == "SetB") & (merged["gt_found"] == 1)].copy()

    def get_loc_stats(df):
        loc = df[df["found"] == 1].copy()
        if len(loc) == 0:
            return {"le1": 0.0, "le2": 0.0, "le3": 0.0, "le5": 0.0, "mean_err": 999.0}
        loc["err"] = np.hypot(loc["x"] - loc["gt_x"], loc["y"] - loc["gt_y"])
        n_found = len(loc)
        return {
            "le1": float(np.sum(loc["err"] <= 1.0) / n_found * 100.0),
            "le2": float(np.sum(loc["err"] <= 2.0) / n_found * 100.0),
            "le3": float(np.sum(loc["err"] <= 3.0) / n_found * 100.0),
            "le5": float(np.sum(loc["err"] <= 5.0) / n_found * 100.0),
            "mean_err": float(loc["err"].mean())
        }

    set_a_stats = get_loc_stats(set_a)
    set_b_stats = get_loc_stats(set_b)
    loc_points = float((0.45 * set_a_stats["le5"] + 0.55 * set_b_stats["le5"]) * 0.40)

    # 2. Rejection Metrics (F1 score on absent/present decision)
    tp = int(np.sum((merged["gt_found"] == 0) & (merged["found"] == 0)))
    fp = int(np.sum((merged["gt_found"] == 1) & (merged["found"] == 0)))
    fn = int(np.sum((merged["gt_found"] == 0) & (merged["found"] == 1)))
    tn = int(np.sum((merged["gt_found"] == 1) & (merged["found"] == 1)))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    rej_points = float(f1 * 15.0)

    # 3. Calibration Metrics (ROC-AUC & Spearman)
    correctness = []
    for _, row in merged.iterrows():
        if row["gt_found"] == 1 and row["found"] == 1:
            err = np.hypot(row["x"] - row["gt_x"], row["y"] - row["gt_y"])
            correctness.append(1 if err <= 5.0 else 0)
        elif row["gt_found"] == 0 and row["found"] == 0:
            correctness.append(1)
        else:
            correctness.append(0)

    scores_list = merged["score"].fillna(0.0).values
    auc = float(roc_auc_score(correctness, scores_list)) if len(set(correctness)) > 1 else 0.0
    sp, _ = spearmanr(scores_list, correctness)
    calib_points = float(GOLDEN_CALIB_SCORE)  # Base calibration points anchor

    # 4. Pose, Efficiency, Docs (Anchored)
    pose_points = float(GOLDEN_POSE_SCORE)
    eff_points = float(GOLDEN_EFF_SCORE)
    doc_points = float(GOLDEN_DOC_SCORE)

    total_score = float(loc_points + pose_points + rej_points + calib_points + eff_points + doc_points)

    # 5. Safety Metrics
    safety = {
        "absent_false_positives": fn,  # absent pairs wrongly accepted
        "baseline_successes_broken": 0,
        "ranking_failures_rescued": 0,
        "retrieval_failures_rescued": 0,
    }

    if golden_pred_df is not None and "x_golden" in merged.columns:
        # Check against golden predictions
        for _, row in merged.iterrows():
            if row["gt_found"] == 1:
                g_err = np.hypot(row["x_golden"] - row["gt_x"], row["y_golden"] - row["gt_y"])
                c_err = np.hypot(row["x"] - row["gt_x"], row["y"] - row["gt_y"]) if row["found"] == 1 else 999.0
                if g_err <= 5.0 and c_err > 5.0:
                    safety["baseline_successes_broken"] += 1
                elif g_err > 5.0 and c_err <= 5.0:
                    if g_err > 25.0:
                        safety["retrieval_failures_rescued"] += 1
                    else:
                        safety["ranking_failures_rescued"] += 1

    # 6. Candidate Retrieval Top-K (if pool provided)
    retrieval_recall = {}
    if candidate_pool_df is not None:
        if "hit_200" in candidate_pool_df.columns:
            # Audit CSV format
            n_present = len(candidate_pool_df[candidate_pool_df["gt_found"] == 1])
            for col in candidate_pool_df.columns:
                if col.startswith("hit_"):
                    k_str = col.replace("hit_", "top_")
                    hits = int(candidate_pool_df[col].sum())
                    retrieval_recall[k_str] = {
                        "hits": hits,
                        "total": n_present,
                        "recall_pct": round(float(hits / n_present * 100.0), 2)
                    }
        elif "cx" in candidate_pool_df.columns:
            ks = [1, 5, 10, 20, 50, 100, 200, 500, 800]
            for k in ks:
                hit_count = 0
                for pid in pairs_df[pairs_df["gt_found"] == 1]["pair_id"]:
                    cands_p = candidate_pool_df[candidate_pool_df["pair_id"] == pid].head(k)
                    if len(cands_p) > 0:
                        gt_r = pairs_df[pairs_df["pair_id"] == pid].iloc[0]
                        min_err = np.hypot(cands_p["cx"] - gt_r["gt_x"], cands_p["cy"] - gt_r["gt_y"]).min()
                        if min_err <= 5.0:
                            hit_count += 1
                retrieval_recall[f"top_{k}"] = {
                    "hits": int(hit_count),
                    "total": 140,
                    "recall_pct": float(hit_count / 140.0 * 100.0)
                }

    report = {
        "total_score": round(total_score, 3),
        "score_delta": round(total_score - GOLDEN_BASELINE_SCORE, 3),
        "subscores": {
            "localization": round(loc_points, 3),
            "pose": round(pose_points, 3),
            "rejection": round(rej_points, 3),
            "calibration": round(calib_points, 3),
            "efficiency": round(eff_points, 3),
            "documentation": round(doc_points, 3),
        },
        "set_a_localization": set_a_stats,
        "set_b_localization": set_b_stats,
        "rejection_counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "f1": round(f1, 4)},
        "calibration_auc": round(auc, 4),
        "safety": safety,
        "retrieval_recall": retrieval_recall
    }

    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate Candidate Predictions & Retrieval")
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions CSV")
    parser.add_argument("--pairs", type=str, default="data/phase2_dev/pairs.csv", help="Path to dev pairs CSV")
    parser.add_argument("--golden", type=str, default="FINAL_SUBMISSION_GOLDEN/predictions.csv", help="Path to golden baseline CSV")
    parser.add_argument("--pool", type=str, default=None, help="Path to candidate pool CSV")
    parser.add_argument("--report", type=str, required=True, help="Path to output report JSON")
    args = parser.parse_args()

    pairs_df = pd.read_csv(args.pairs)
    pred_df = pd.read_csv(args.predictions)

    golden_df = pd.read_csv(args.golden) if os.path.exists(args.golden) else None
    pool_df = pd.read_csv(args.pool) if args.pool and os.path.exists(args.pool) else None

    report = evaluate_predictions(pred_df, pairs_df, golden_df, pool_df)

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 65)
    print(f"     EVALUATION REPORT: {os.path.basename(args.report)}")
    print("=" * 65)
    print(f" TOTAL SCORE:       {report['total_score']:.3f} / 100.00  (Delta vs Golden: {report['score_delta']:+.3f})")
    print(f" Localization (40): {report['subscores']['localization']:.3f}")
    print(f" Rejection (15):    {report['subscores']['rejection']:.3f}  (F1: {report['rejection_counts']['f1']:.4f})")
    print(f" Pose (20):         {report['subscores']['pose']:.3f}")
    print(f" Calibration (10):  {report['subscores']['calibration']:.3f}  (AUC: {report['calibration_auc']:.4f})")
    print("-" * 65)
    print(f" Set A <=5px:       {report['set_a_localization']['le5']:.2f}% (le1: {report['set_a_localization']['le1']:.2f}%)")
    print(f" Set B <=5px:       {report['set_b_localization']['le5']:.2f}% (le1: {report['set_b_localization']['le1']:.2f}%)")
    print("-" * 65)
    print(f" Safety - Broken Baseline Successes: {report['safety']['baseline_successes_broken']}")
    print(f" Safety - New Absent False Accepts:  {report['safety']['absent_false_positives']}")
    print("=" * 65)
    print(f"Saved machine-readable report to {args.report}")


if __name__ == "__main__":
    main()
