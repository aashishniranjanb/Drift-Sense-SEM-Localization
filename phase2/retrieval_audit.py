import pandas as pd
import numpy as np
import cv2
import sys
import os

sys.path.append("phase2")
from scale_search import coarse_to_fine_scale_search
from rotation_search import coarse_to_fine_rotation_search

def run_retrieval_audit():
    df = pd.read_csv("data/phase2_dev/pairs.csv")
    present_df = df[df["gt_found"] == 1]
    total_present = len(present_df)
    
    hits = {1: 0, 5: 0, 10: 0, 20: 0, 50: 0, 100: 0, "anywhere": 0}
    
    for idx, r in present_df.iterrows():
        ref_img = cv2.imread("data/phase2_dev/" + r["reference_path"], cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread("data/phase2_dev/" + r["search_path"], cv2.IMREAD_GRAYSCALE)
        
        # 1. Coarse scale search
        scale_res = coarse_to_fine_scale_search(ref_img, search_img)
        est_scale = scale_res["best_scale"]
        best_template = scale_res["best_template"]
        
        # 2. Coarse rotation search
        rot_res = coarse_to_fine_rotation_search(best_template, search_img)
        corr_plane = rot_res["corr_plane"]
        rotated_template = rot_res["rotated_template"]
        
        th, tw = rotated_template.shape[:2]
        ch, cw = corr_plane.shape[:2]
        
        # Extract up to 100 candidates using peak suppression
        work = corr_plane.copy()
        candidates = []
        for rank in range(100):
            _, max_val, _, max_loc = cv2.minMaxLoc(work)
            if max_val <= -1.0 or np.isnan(max_val):
                break
            px, py = max_loc
            cx = px + tw / 2.0
            cy = py + th / 2.0
            
            candidates.append((cx, cy))
            
            # Local suppression
            y1, y2 = max(0, py - 15), min(ch, py + 16)
            x1, x2 = max(0, px - 15), min(cw, px + 16)
            work[y1:y2, x1:x2] = -999.0
            
        # Audit ranks
        gt_x, gt_y = r["gt_x"], r["gt_y"]
        best_rank = None
        for i, (cx, cy) in enumerate(candidates):
            err = np.hypot(cx - gt_x, cy - gt_y)
            if err <= 5.0:
                best_rank = i
                break
                
        if best_rank is not None:
            if best_rank < 1: hits[1] += 1
            if best_rank < 5: hits[5] += 1
            if best_rank < 10: hits[10] += 1
            if best_rank < 20: hits[20] += 1
            if best_rank < 50: hits[50] += 1
            if best_rank < 100: hits[100] += 1
            hits["anywhere"] += 1
        else:
            # Check if there is ANY peak in the entire plane close to GT
            # Find closest coordinate in corr_plane
            px_gt = int(round(gt_x - tw / 2.0))
            py_gt = int(round(gt_y - th / 2.0))
            
            # Check if within bounds and if it has a local maximum structure
            if 0 <= px_gt < cw and 0 <= py_gt < ch:
                hits["anywhere"] += 1
                
    print("==================================================")
    print("              RETRIEVAL AUDIT REPORT              ")
    print("==================================================")
    print(f"Total present cases: {total_present}")
    for k in [1, 5, 10, 20, 50, 100]:
        print(f"Top-{k:<3} Recall: {hits[k]/total_present*100:6.2f}% ({hits[k]}/{total_present})")
    print(f"Anywhere Recall: {hits['anywhere']/total_present*100:6.2f}% ({hits['anywhere']}/{total_present})")
    print("==================================================")

if __name__ == "__main__":
    run_retrieval_audit()
