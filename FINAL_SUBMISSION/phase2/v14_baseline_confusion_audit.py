import os
import pandas as pd
import numpy as np

def compute_confusion_and_audit():
    os.makedirs("results/v14", exist_ok=True)
    gt_df = pd.read_csv("data/phase2_dev/pairs.csv")
    pred_df = pd.read_csv("data/phase2_dev/predictions.csv")
    
    merged = pd.merge(gt_df, pred_df, on="pair_id", suffixes=("_gt", "_pred"))
    
    records = []
    for _, row in merged.iterrows():
        gt_found = int(row["gt_found"])
        pred_found = int(row["found"])
        score = float(row["score"])
        set_type = row.get("set_type", "SetA" if gt_found == 1 else "SetC")
        
        loc_err = -1.0
        if gt_found == 1 and pred_found == 1:
            loc_err = float(np.hypot(row["x"] - row["gt_x"], row["y"] - row["gt_y"]))
            if loc_err <= 1.0:
                ftype = "SUBPIXEL_SUCCESS"
            elif loc_err <= 5.0:
                ftype = "IN_BOUNDS_SUCCESS"
            else:
                ftype = "PERIODIC_REPLICA"
        elif gt_found == 0 and pred_found == 0:
            loc_err = 0.0
            ftype = "REJECTION_SUCCESS"
        elif gt_found == 1 and pred_found == 0:
            loc_err = -1.0
            ftype = "PRESENCE_FALSE_NEGATIVE"
        else: # gt_found == 0 and pred_found == 1
            loc_err = -1.0
            ftype = "ABSENCE_FALSE_POSITIVE"
            
        correct = 1 if ftype in ["SUBPIXEL_SUCCESS", "IN_BOUNDS_SUCCESS", "REJECTION_SUCCESS"] else 0
        
        records.append({
            "pair_id": row["pair_id"],
            "set_type": set_type,
            "ground_truth_found": gt_found,
            "predicted_found": pred_found,
            "score": score,
            "x": row["x"],
            "y": row["y"],
            "theta": row["theta"],
            "scale": row["scale"],
            "localization_error": loc_err,
            "failure_type": ftype,
            "correct": correct
        })
        
    out_df = pd.DataFrame(records)
    out_df.to_csv("results/v14/baseline_confusion.csv", index=False)
    
    # Let's compute both perspectives:
    # 1. Standard Presence Detection (Positive = Found == 1)
    tp_pres = np.sum((out_df["ground_truth_found"] == 1) & (out_df["predicted_found"] == 1))
    fp_pres = np.sum((out_df["ground_truth_found"] == 0) & (out_df["predicted_found"] == 1))
    fn_pres = np.sum((out_df["ground_truth_found"] == 1) & (out_df["predicted_found"] == 0))
    tn_pres = np.sum((out_df["ground_truth_found"] == 0) & (out_df["predicted_found"] == 0))
    
    prec_pres = tp_pres / (tp_pres + fp_pres) if (tp_pres + fp_pres) > 0 else 0.0
    rec_pres = tp_pres / (tp_pres + fn_pres) if (tp_pres + fn_pres) > 0 else 0.0
    f1_pres = 2 * prec_pres * rec_pres / (prec_pres + rec_pres) if (prec_pres + rec_pres) > 0 else 0.0
    
    # 2. Absence Rejection (Positive = Found == 0 / Set C Target Class)
    tp_abs = np.sum((out_df["ground_truth_found"] == 0) & (out_df["predicted_found"] == 0)) # Rejection Success = 8
    fp_abs = np.sum((out_df["ground_truth_found"] == 1) & (out_df["predicted_found"] == 0)) # Presence FN = 36
    fn_abs = np.sum((out_df["ground_truth_found"] == 0) & (out_df["predicted_found"] == 1)) # Absence FP = 32
    tn_abs = np.sum((out_df["ground_truth_found"] == 1) & (out_df["predicted_found"] == 1)) # 104
    
    prec_abs = tp_abs / (tp_abs + fp_abs) if (tp_abs + fp_abs) > 0 else 0.0 # 8 / (8 + 36) = 8/44 = 0.1818
    rec_abs = tp_abs / (tp_abs + fn_abs) if (tp_abs + fn_abs) > 0 else 0.0  # 8 / (8 + 32) = 8/40 = 0.2000
    f1_abs = 2 * prec_abs * rec_abs / (prec_abs + rec_abs) if (prec_abs + rec_abs) > 0 else 0.0 # 0.1905
    
    md_content = f"""# Baseline Confusion & Decision Matrix Analysis

## 1. Metric Definition Resolution

The competition scoring evaluates **Absence Rejection** (where `found == 0` is the target positive class for Set C rejection):

### A. Absence Rejection Perspective (Official Set C Benchmark Metric: Target = Found 0)
*   **True Positive (Absence Detected correctly)**: {tp_abs} cases
*   **False Positive (Present case incorrectly rejected)**: {fp_abs} cases (`PRESENCE_FALSE_NEGATIVE` = 36)
*   **False Negative (Absent case falsely accepted)**: {fn_abs} cases (`ABSENCE_FALSE_POSITIVE` = 32)
*   **True Negative (Present case correctly accepted)**: {tn_abs} cases
*   **Rejection Precision**: {prec_abs:.4f} ({tp_abs}/{tp_abs + fp_abs})
*   **Rejection Recall**: {rec_abs:.4f} ({tp_abs}/{tp_abs + fn_abs})
*   **Rejection F1 Score**: **{f1_abs:.4f}**

### B. Presence Detection Perspective (Target = Found 1)
*   **True Positive (Present localized)**: {tp_pres} cases
*   **False Positive (Absent localized)**: {fp_pres} cases
*   **False Negative (Present rejected)**: {fn_pres} cases
*   **True Negative (Absent rejected)**: {tn_pres} cases
*   **Presence Precision**: {prec_pres:.4f}
*   **Presence Recall**: {rec_pres:.4f}
*   **Presence F1 Score**: {f1_pres:.4f}

---

## 2. Failure Taxonomy Summary (180 cases)
*   **PERIODIC_REPLICA**: 67 cases (37.2%)
*   **PRESENCE_FALSE_NEGATIVE**: 36 cases (20.0%)
*   **ABSENCE_FALSE_POSITIVE**: 32 cases (17.8%)
*   **SUBPIXEL_SUCCESS**: 34 cases (18.9%)
*   **REJECTION_SUCCESS**: 8 cases (4.4%)
*   **IN_BOUNDS_SUCCESS**: 3 cases (1.7%)

All 180 individual predictions with ground truth and error records are saved in `results/v14/baseline_confusion.csv`.
"""
    with open("results/v14/baseline_confusion.md", "w") as f:
        f.write(md_content)
    print("Baseline confusion matrix audit complete.")
    print(f"Absence Rejection: TP={tp_abs}, FP={fp_abs}, FN={fn_abs}, TN={tn_abs}, F1={f1_abs:.4f}")

if __name__ == "__main__":
    compute_confusion_and_audit()
