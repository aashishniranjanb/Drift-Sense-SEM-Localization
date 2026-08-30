import os
import argparse
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

def evaluate_adversarial(input_csv, predictions_csv, output_dir=None):
    gt = pd.read_csv(input_csv)
    pred = pd.read_csv(predictions_csv)
    
    # Merge on pair_id
    merged = pd.merge(gt, pred, on="pair_id", suffixes=("_gt", "_pred"))
    
    if output_dir is None:
        output_dir = os.path.dirname(predictions_csv)
        
    categories = merged["set_type"].unique()
    
    reports = []
    
    print("=========================================================================================")
    print("                      ADVERSARIAL STRESS-TEST BENCHMARK REPORT                          ")
    print("=========================================================================================")
    print(f"{'Category':<32} | {'Loc <=5px':<10} | {'Scale MAE':<10} | {'Rot MAE':<10} | {'Rej F1':<8} | {'Spearman':<8}")
    print("-----------------------------------------------------------------------------------------")
    
    for cat in sorted(categories):
        cat_data = merged[merged["set_type"] == cat]
        
        # 1. Localization Metric (<= 5px)
        present_gt = cat_data[cat_data["gt_found"] == 1]
        localized = 0.0
        scale_mae = 0.0
        rot_mae = 0.0
        
        if len(present_gt) > 0:
            present_pred = present_gt[present_gt["found"] == 1]
            if len(present_pred) > 0:
                errs = np.hypot(present_pred["x"] - present_pred["gt_x"], present_pred["y"] - present_pred["gt_y"])
                localized = (np.sum(errs <= 5.0) / len(present_gt)) * 100.0
                
            scale_mae = np.mean(np.abs(present_gt["scale"] - present_gt["gt_scale"]))
            rot_mae = np.mean(np.abs(present_gt["theta"] - present_gt["gt_theta"]))
            
        # 2. Rejection Metric (F1 Score)
        absent_gt = cat_data[cat_data["gt_found"] == 0]
        rej_f1 = 0.0
        
        # Calculate Precision/Recall for absent class (class 0)
        # Class 0 prediction is found == 0
        total_pred_absent = len(cat_data[cat_data["found"] == 0])
        total_gt_absent = len(absent_gt)
        
        tp = len(cat_data[(cat_data["gt_found"] == 0) & (cat_data["found"] == 0)])
        
        precision = tp / total_pred_absent if total_pred_absent > 0 else 0.0
        recall = tp / total_gt_absent if total_gt_absent > 0 else 0.0
        rej_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        # 3. Spearman Rank Correlation
        # Compute correctness vs score
        correctness = []
        scores = []
        for _, row in cat_data.iterrows():
            gt_f = int(row["gt_found"])
            pred_f = int(row["found"])
            correct = 0
            if gt_f == 1 and pred_f == 1:
                err = np.hypot(row["x"] - row["gt_x"], row["y"] - row["gt_y"])
                if err <= 5.0:
                    correct = 1
            elif gt_f == 0 and pred_f == 0:
                correct = 1
            correctness.append(correct)
            scores.append(row["score"])
            
        rho = 0.0
        if len(np.unique(correctness)) > 1:
            rho, _ = spearmanr(correctness, scores)
            if np.isnan(rho):
                rho = 0.0
                
        print(f"{cat:<32} | {localized:<9.1f}% | {scale_mae:<10.4f} | {rot_mae:<10.4f} | {rej_f1:<8.4f} | {rho:<8.4f}")
        
        reports.append({
            "category": cat,
            "loc_5px": localized,
            "scale_mae": scale_mae,
            "rot_mae": rot_mae,
            "rejection_f1": rej_f1,
            "spearman": rho
        })
        
    print("=========================================================================================")
    
    # Save markdown summary
    md_lines = [
        "# Adversarial Stress-Test Benchmark Baseline\n",
        "| Category | Localization <= 5px (%) | Scale MAE | Rotation MAE | Rejection F1 | Spearman $\\rho$ |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |"
    ]
    for r in reports:
        md_lines.append(f"| {r['category']} | {r['loc_5px']:.1f}% | {r['scale_mae']:.4f} | {r['rot_mae']:.4f} | {r['rejection_f1']:.4f} | {r['spearman']:.4f} |")
        
    with open(os.path.join(output_dir, "V10.1_ADVERSARIAL_BASELINE.md"), "w") as f:
        f.write("\n".join(md_lines))
    print(f"Archived report to {os.path.join(output_dir, 'V10.1_ADVERSARIAL_BASELINE.md')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--predictions-csv", required=True)
    args = parser.parse_args()
    evaluate_adversarial(args.input_csv, args.predictions_csv)
