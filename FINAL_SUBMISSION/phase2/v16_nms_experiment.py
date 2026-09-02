import os
import sys
import cv2
import numpy as np
import pandas as pd
import time

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, "phase2"))
sys.path.append(os.path.join(parent_dir, "fallbacks"))
from inference_phase2 import compute_psr

def extract_r0(corr_plane, tw, th, max_k=50):
    # R0: Current (r=5)
    return extract_nms(corr_plane, tw, th, max_k, r=5)

def extract_nms(corr_plane, tw, th, max_k, r):
    ch, cw = corr_plane.shape[:2]
    work = corr_plane.copy()
    cands = []
    for rank in range(max_k):
        _, max_val, _, max_loc = cv2.minMaxLoc(work)
        if max_val <= 0.01 or np.isnan(max_val): break
        px, py = max_loc
        cands.append({
            "peak_x": px, "peak_y": py,
            "cx": px + tw / 2.0, "cy": py + th / 2.0,
            "corr_score": float(max_val),
            "raw_rank": rank + 1
        })
        y1, y2 = max(0, py - r), min(ch, py + r + 1)
        x1, x2 = max(0, px - r), min(cw, px + r + 1)
        work[y1:y2, x1:x2] = -999.0
    return cands

def extract_r1(corr_plane, tw, th, max_k=50):
    # R1: Less aggressive (r=2)
    return extract_nms(corr_plane, tw, th, max_k, r=2)

def extract_r2(corr_plane, tw, th, max_k=50):
    # R2: Adaptive NMS. Calculate PSR of the global max. 
    # If PSR is low (periodic), use r=3. If high (unique), use r=10 to clear out junk.
    psr, _, _ = compute_psr(corr_plane, -1, -1)
    r = 3 if psr < 10.0 else 10
    return extract_nms(corr_plane, tw, th, max_k, r=r)

def extract_r3_family(corr_plane, tw, th, max_k=50, ref_img=None, search_img=None, est_scale=10.0, est_theta=0.0):
    # R3: Bounded Rescue Queue based on Context
    # Extract 200 candidates with fast NMS
    pool = extract_nms(corr_plane, tw, th, max_k=200, r=5)
    if len(pool) <= max_k:
        return pool
        
    if ref_img is None:
        return pool[:max_k]
        
    from inference_phase2 import verify_candidate_context
    
    # Compute context for all 200
    for c in pool:
        ctx_res = verify_candidate_context(ref_img, search_img, c["cx"], c["cy"], est_scale, est_theta)
        c["rescue_score"] = c["corr_score"] + 0.1 * ctx_res["combined"]  # Lightweight blend
        
    # Sort by rescue score
    pool.sort(key=lambda x: x["rescue_score"], reverse=True)
    return pool[:max_k]

def test_extractors():
    # We will test using V15_ORACLE_AUDIT data, but actually we need to run on the images!
    # Let's load the pairs and run pose fallback, then the extractor, and see if GT is in Top-50.
    from fallbacks.pose_fallback import perform_pose_fallback_search
    
    df_pairs = pd.read_csv("data/phase2_dev/pairs.csv")
    present_df = df_pairs[df_pairs["gt_found"] == 1].head(30) # Test on first 30 cases for speed
    
    print(f"Testing Extractors on {len(present_df)} cases...")
    results = {"R0": 0, "R1": 0, "R2": 0, "R3": 0}
    
    for idx, row in present_df.iterrows():
        pair_id = row["pair_id"]
        ref_path = os.path.join("data/phase2_dev", row["reference_path"])
        search_path = os.path.join("data/phase2_dev", row["search_path"])
        gt_x, gt_y = float(row["gt_x"]), float(row["gt_y"])
        
        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
        
        # Get raw correlation plane from fallback pose
        pose_res = perform_pose_fallback_search(ref_img, search_img)
        corr_plane = pose_res["corr_plane"]
        rotated_template = pose_res["best_template"]
        th, tw = rotated_template.shape[:2]
        
        def check_gt_in_cands(cands, thresh=5.0):
            for c in cands:
                if np.hypot(c["cx"] - gt_x, c["cy"] - gt_y) <= thresh:
                    return True
            return False
            
        cands_r0 = extract_r0(corr_plane, tw, th, 50)
        cands_r1 = extract_r1(corr_plane, tw, th, 50)
        cands_r2 = extract_r2(corr_plane, tw, th, 50)
        
        est_scale = pose_res["best_scale"]
        est_theta = pose_res["best_theta"]
        cands_r3 = extract_r3_family(corr_plane, tw, th, 50, ref_img, search_img, est_scale, est_theta)
        
        if check_gt_in_cands(cands_r0): results["R0"] += 1
        if check_gt_in_cands(cands_r1): results["R1"] += 1
        if check_gt_in_cands(cands_r2): results["R2"] += 1
        if check_gt_in_cands(cands_r3): results["R3"] += 1
        
        print(f"Case {pair_id} | R0: {results['R0']}, R1: {results['R1']}, R2: {results['R2']}, R3: {results['R3']}")
        
    print("\nFINAL RESULTS (Top-50 Recovery on 30 cases):")
    for k, v in results.items():
        print(f"{k}: {v} / 30 ({(v/30)*100:.2f}%)")

if __name__ == '__main__':
    test_extractors()
