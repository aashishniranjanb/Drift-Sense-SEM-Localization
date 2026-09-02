import argparse
import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

def evaluate_phase2(input_csv, predictions_csv, output_dir=None):
    gt = pd.read_csv(input_csv)
    pred = pd.read_csv(predictions_csv)
    
    # Merge on pair_id
    merged = pd.merge(gt, pred, on="pair_id", suffixes=("_gt", "_pred"))
    
    if output_dir is None:
        output_dir = os.path.dirname(predictions_csv)
        
    taxonomy_records = []
    
    # Per-Set accumulators
    sets_data = {"SetA": [], "SetB": [], "SetC": []}
    
    for idx, row in merged.iterrows():
        set_type = row.get("set_type", "SetA" if row["gt_found"] == 1 else "SetC")
        gt_found = int(row["gt_found"])
        pred_found = int(row["found"])
        
        scale_err = abs(row["scale"] - row["gt_scale"]) if gt_found == 1 else 0.0
        theta_err = abs(row["theta"] - row["gt_theta"]) if gt_found == 1 else 0.0
        
        if gt_found == 1 and pred_found == 1:
            loc_err = float(np.hypot(row["x"] - row["gt_x"], row["y"] - row["gt_y"]))
            if loc_err <= 1.0:
                failure_mode = "SUBPIXEL_SUCCESS"
            elif loc_err <= 5.0:
                failure_mode = "IN_BOUNDS_SUCCESS"
            else:
                failure_mode = "PERIODIC_REPLICA"
        elif gt_found == 0 and pred_found == 0:
            loc_err = 0.0
            failure_mode = "REJECTION_SUCCESS"
        elif gt_found == 1 and pred_found == 0:
            loc_err = -1.0
            failure_mode = "PRESENCE_FALSE_NEGATIVE"
        else: # gt_found == 0 and pred_found == 1
            loc_err = -1.0
            failure_mode = "ABSENCE_FALSE_POSITIVE"
            
        rec = {
            "pair_id": row["pair_id"],
            "set_type": set_type,
            "gt_found": gt_found,
            "pred_found": pred_found,
            "loc_err": loc_err,
            "scale_err": scale_err,
            "theta_err": theta_err,
            "score": row["score"],
            "failure_mode": failure_mode
        }
        
        taxonomy_records.append(rec)
        if set_type in sets_data:
            sets_data[set_type].append(rec)
            
    df_tax = pd.DataFrame(taxonomy_records)
    tax_csv = os.path.join(output_dir, "failure_taxonomy.csv")
    df_tax.to_csv(tax_csv, index=False)
    
    # Analyze metrics per Set
    def compute_set_metrics(records):
        if len(records) == 0:
            return {"count": 0, "le_1": 0.0, "le_5": 0.0, "median": 0.0, "scale_mae": 0.0, "theta_mae": 0.0, "f1": 0.0}
        df_r = pd.DataFrame(records)
        count = len(df_r)
        
        present_gt = df_r[df_r["gt_found"] == 1]
        if len(present_gt) > 0:
            localized = present_gt[present_gt["pred_found"] == 1]
            errs = localized["loc_err"].values
            le_1 = np.mean(errs <= 1.0) * 100.0 if len(errs) > 0 else 0.0
            le_5 = np.mean(errs <= 5.0) * 100.0 if len(errs) > 0 else 0.0
            median = np.median(errs) if len(errs) > 0 else 0.0
            scale_mae = np.mean(localized["scale_err"].values) if len(localized) > 0 else 0.0
            theta_mae = np.mean(localized["theta_err"].values) if len(localized) > 0 else 0.0
        else:
            le_1 = le_5 = median = scale_mae = theta_mae = 0.0
            
        tp = np.sum((df_r["gt_found"] == 1) & (df_r["pred_found"] == 1))
        fp = np.sum((df_r["gt_found"] == 0) & (df_r["pred_found"] == 1))
        fn = np.sum((df_r["gt_found"] == 1) & (df_r["pred_found"] == 0))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "count": count,
            "le_1": le_1,
            "le_5": le_5,
            "median": median,
            "scale_mae": scale_mae,
            "theta_mae": theta_mae,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    mA = compute_set_metrics(sets_data["SetA"])
    mB = compute_set_metrics(sets_data["SetB"])
    mC = compute_set_metrics(sets_data["SetC"])
    
    # Rejection F1 score computed across the entire dataset (where found == 0 is the target positive class)
    tp_rej = np.sum((df_tax["gt_found"] == 0) & (df_tax["pred_found"] == 0))
    fp_rej = np.sum((df_tax["gt_found"] == 1) & (df_tax["pred_found"] == 0))
    fn_rej = np.sum((df_tax["gt_found"] == 0) & (df_tax["pred_found"] == 1))
    
    precision_rej = tp_rej / (tp_rej + fp_rej) if (tp_rej + fp_rej) > 0 else 0.0
    recall_rej = tp_rej / (tp_rej + fn_rej) if (tp_rej + fn_rej) > 0 else 0.0
    f1_rej = 2 * precision_rej * recall_rej / (precision_rej + recall_rej) if (precision_rej + recall_rej) > 0 else 0.0
    
    # Official weighted localization score
    weighted_loc_score = 0.45 * mA["le_5"] + 0.55 * mB["le_5"]
    
    # Spearman rank correlation
    correctness = []
    scores = []
    for r in taxonomy_records:
        is_corr = 1 if (r["failure_mode"] in ["SUBPIXEL_SUCCESS", "IN_BOUNDS_SUCCESS", "REJECTION_SUCCESS"]) else 0
        correctness.append(is_corr)
        scores.append(r["score"])
    spearman_corr, _ = spearmanr(scores, correctness)
    if np.isnan(spearman_corr):
        spearman_corr = 0.0
        
    print("\n" + "="*65)
    print("           DRIFT-SENSE++ PHASE 2 HARDENED BENCHMARK")
    print("="*65)
    print(f"Total Evaluated Pairs: {len(merged)}")
    print(f"  - Set A (Nominal):   {mA['count']}")
    print(f"  - Set B (Degraded):  {mB['count']}")
    print(f"  - Set C (Absent):    {mC['count']}")
    print("-" * 65)
    print("1. LOCALIZATION METRICS (<= 5 px Target):")
    print(f"  Set A <= 1 px: {mA['le_1']:.2f}% | Set A <= 5 px: {mA['le_5']:.2f}% | Median: {mA['median']:.2f} px")
    print(f"  Set B <= 1 px: {mB['le_1']:.2f}% | Set B <= 5 px: {mB['le_5']:.2f}% | Median: {mB['median']:.2f} px")
    print(f"  OFFICIAL WEIGHTED LOC SCORE (0.45*A + 0.55*B): {weighted_loc_score:.2f}%")
    print("-" * 65)
    print("2. POSE RECOVERY METRICS:")
    print(f"  Set A Scale MAE: {mA['scale_mae']:.4f} | Rotation MAE: {mA['theta_mae']:.4f}°")
    print(f"  Set B Scale MAE: {mB['scale_mae']:.4f} | Rotation MAE: {mB['theta_mae']:.4f}°")
    print("-" * 65)
    print("3. ABSENCE REJECTION METRICS (Set C Target F1 > 0.90):")
    print(f"  Overall Precision: {precision_rej:.4f} | Recall: {recall_rej:.4f}")
    print(f"  Set C Rejection F1 Score: {f1_rej:.4f}")
    print("-" * 65)
    print("4. CONFIDENCE MONOTONICITY:")
    print(f"  Spearman Rank Correlation (rho): {spearman_corr:.4f}")
    print("-" * 65)
    print("5. FAILURE TAXONOMY SUMMARY:")
    tax_counts = df_tax["failure_mode"].value_counts().to_dict()
    for k, v in tax_counts.items():
        print(f"  - {k}: {v} cases ({v/len(df_tax)*100.0:.1f}%)")
    print("="*65 + "\n")
    print(f"Failure taxonomy written to {tax_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 Benchmark Harness")
    parser.add_argument("--input-csv", required=True, help="Path to ground truth CSV")
    parser.add_argument("--predictions-csv", required=True, help="Path to predictions CSV")
    args = parser.parse_args()
    evaluate_phase2(args.input_csv, args.predictions_csv)
