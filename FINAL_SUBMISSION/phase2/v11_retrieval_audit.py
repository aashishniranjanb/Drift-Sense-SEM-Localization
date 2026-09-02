import pandas as pd
import numpy as np
import cv2
import sys
import os
from scipy.ndimage import maximum_filter

sys.path.append("phase2")
from scale_search import coarse_to_fine_scale_search
from rotation_search import coarse_to_fine_rotation_search
from candidate_ranker import rank_candidates
from pose_refinement import refine_pose

from channel_consensus import extract_gradient
from adaptive_peak_detector import extract_adaptive_candidates

def local_maxima_detector(corr, w=4):
    size = 2 * w + 1
    local_max = (maximum_filter(corr, size=size) == corr)
    local_max = local_max & (corr > 0.01)
    y_indices, x_indices = np.where(local_max)
    scores = corr[y_indices, x_indices]
    sorted_idx = np.argsort(scores)[::-1]
    return list(zip(x_indices[sorted_idx], y_indices[sorted_idx], scores[sorted_idx]))

def run_retrieval_audit():
    df = pd.read_csv("data/phase2_dev/pairs.csv")
    present_df = df[df["gt_found"] == 1]
    total_present = len(present_df)
    
    audit_records = []
    
    for idx, r in present_df.iterrows():
        ref_img = cv2.imread("data/phase2_dev/" + r["reference_path"], cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread("data/phase2_dev/" + r["search_path"], cv2.IMREAD_GRAYSCALE)
        
        # Coarse Scale and Rotation
        scale_res = coarse_to_fine_scale_search(ref_img, search_img)
        est_scale = scale_res["best_scale"]
        best_template = scale_res["best_template"]
        
        rot_res = coarse_to_fine_rotation_search(best_template, search_img)
        corr = rot_res["corr_plane"]
        rotated_template = rot_res["rotated_template"]
        
        th, tw = rotated_template.shape[:2]
        ch, cw = corr.shape[:2]
        
        # 1. Check if GT peak exists in correlation plane
        gt_x, gt_y = r["gt_x"], r["gt_y"]
        gt_px = int(round(gt_x - tw / 2.0))
        gt_py = int(round(gt_y - th / 2.0))
        
        raw_plane_available = False
        if 0 <= gt_px < cw and 0 <= gt_py < ch:
            raw_plane_available = True
            
        # 2. Extract all local maxima to find raw_rank of the GT
        peaks = local_maxima_detector(corr, w=4)
        raw_rank = -1
        gt_corr_score = 0.0
        for rank, (px, py, val) in enumerate(peaks):
            cx, cy = px + tw / 2.0, py + th / 2.0
            if np.hypot(cx - gt_x, cy - gt_y) <= 5.0:
                raw_rank = rank
                gt_corr_score = val
                break
                
        # 3. Simulate adaptive candidate pool extraction (Top-20 V11.1)
        search_grad = extract_gradient(search_img)
        rotated_template_grad = extract_gradient(rotated_template)
        corr_grad = cv2.matchTemplate(search_grad, rotated_template_grad, cv2.TM_CCOEFF_NORMED)
        
        candidates_std = extract_adaptive_candidates(corr, corr_grad, tw, th, max_k=50)
            
        # Check recall at various levels
        top1 = False
        top5 = False
        top10 = False
        top20 = False
        top50 = False
        
        distance_to_top1 = 999.0
        distance_to_top5 = 999.0
        distance_to_top20 = 999.0
        distance_to_top50 = 999.0
        
        for i, c in enumerate(candidates_std):
            err = np.hypot(c["cx"] - gt_x, c["cy"] - gt_y)
            if err <= 5.0:
                if i < 1: top1 = True
                if i < 5: top5 = True
                if i < 10: top10 = True
                if i < 20: top20 = True
                if i < 50: top50 = True
            if i == 0: distance_to_top1 = err
            if i < 5: distance_to_top5 = min(distance_to_top5, err)
            if i < 20: distance_to_top20 = min(distance_to_top20, err)
            if i < 50: distance_to_top50 = min(distance_to_top50, err)
            
        # Classifier failure categorization
        # A = GT never exists in correlation plane
        # B = GT exists but peak extraction misses it
        # C = GT retrieved but replica outranks it
        # D = GT correctly ranked but metrology refinement fails
        # E = present/absent decision failure (e.g. found=0 when it should be 1)
        failure_mode = "C"  # Default
        if not raw_plane_available or raw_rank == -1:
            failure_mode = "A"
        elif not top50:
            failure_mode = "B"
        else:
            # GT is in Top-50
            # Check if ranked #1
            if top1:
                # Refine pose
                rx, ry, _, _ = refine_pose(ref_img, search_img, est_scale, rot_res["best_theta"], candidates_std[0]["peak_x"], candidates_std[0]["peak_y"], corr)
                if np.hypot(rx - gt_x, ry - gt_y) <= 5.0:
                    failure_mode = "SUCCESS"
                else:
                    failure_mode = "D"
            else:
                failure_mode = "C"
                
        audit_records.append({
            "pair_id": r["pair_id"],
            "gt_x": gt_x,
            "gt_y": gt_y,
            "raw_plane_available": int(raw_plane_available),
            "raw_rank": raw_rank,
            "top1": int(top1),
            "top5": int(top5),
            "top10": int(top10),
            "top20": int(top20),
            "top50": int(top50),
            "distance_to_top1": distance_to_top1,
            "distance_to_top5": distance_to_top5,
            "distance_to_top20": distance_to_top20,
            "distance_to_top50": distance_to_top50,
            "failure_mode": failure_mode
        })
        
    audit_df = pd.DataFrame(audit_records)
    os.makedirs("results/phase2/V11_MAIN_TRACK", exist_ok=True)
    audit_df.to_csv("results/phase2/V11_MAIN_TRACK/V11_RETRIEVAL_RESULTS.csv", index=False)
    
    # Save failure taxonomy summary
    tax_counts = audit_df["failure_mode"].value_counts().to_dict()
    for mode in ["A", "B", "C", "D", "E", "SUCCESS"]:
        if mode not in tax_counts:
            tax_counts[mode] = 0
            
    tax_df = pd.DataFrame([{"mode": m, "count": tax_counts[m]} for m in tax_counts])
    tax_df.to_csv("results/phase2/V11_MAIN_TRACK/V11_FAILURE_TAXONOMY.csv", index=False)
    
    # Print summary
    print("==================================================")
    print("           V11 SYSTEMATIC RETRIEVAL AUDIT         ")
    print("==================================================")
    print(f"Total evaluated pairs: {total_present}")
    print(f"A (GT never in corr plane):       {tax_counts['A']} ({tax_counts['A']/total_present*100:.2f}%)")
    print(f"B (GT exists but NMS missed it):   {tax_counts['B']} ({tax_counts['B']/total_present*100:.2f}%)")
    print(f"C (GT retrieved but outranked):   {tax_counts['C']} ({tax_counts['C']/total_present*100:.2f}%)")
    print(f"D (GT ranked #1 but ref. failed):  {tax_counts['D']} ({tax_counts['D']/total_present*100:.2f}%)")
    print(f"SUCCESS:                          {tax_counts['SUCCESS']} ({tax_counts['SUCCESS']/total_present*100:.2f}%)")
    print("==================================================")
    
    # Write V11_RETRIEVAL_AUDIT.md
    with open("results/phase2/V11_MAIN_TRACK/V11_RETRIEVAL_AUDIT.md", "w") as f:
        f.write(f"""# V11 Systematic Retrieval Audit

## Failure Taxonomy counts
- **A (GT never in correlation plane)**: {tax_counts['A']} ({tax_counts['A']/total_present*100:.2f}%)
- **B (GT exists but NMS peak extraction missed it)**: {tax_counts['B']} ({tax_counts['B']/total_present*100:.2f}%)
- **C (GT retrieved but wrong replica outranked it)**: {tax_counts['C']} ({tax_counts['C']/total_present*100:.2f}%)
- **D (GT correctly ranked #1 but pose refinement failed)**: {tax_counts['D']} ({tax_counts['D']/total_present*100:.2f}%)
- **SUCCESS**: {tax_counts['SUCCESS']} ({tax_counts['SUCCESS']/total_present*100:.2f}%)
""")
        
if __name__ == "__main__":
    run_retrieval_audit()
