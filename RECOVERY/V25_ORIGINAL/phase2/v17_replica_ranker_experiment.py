import os
import sys
import cv2
import numpy as np
import pandas as pd

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, "phase2"))
sys.path.append(os.path.join(parent_dir, "fallbacks"))
sys.path.append(os.path.join(parent_dir, "team", "akhilesh-localization"))

from fallbacks.pose_fallback import perform_pose_fallback_search
from candidate_extractor import extract_candidates_akhilesh
from family_clustering import cluster_replica_families
from spatial_fingerprint import compute_spatial_fingerprint
from context_matcher import verify_candidate_context
from inference_phase2 import (
    verify_phase_consistency,
    compute_psr,
    estimator_a_phase_correlation
)

def evaluate_ranker_variants():
    pairs_df = pd.read_csv("data/phase2_dev/pairs.csv")
    present_df = pairs_df[pairs_df["gt_found"] == 1].copy()
    
    print(f"Evaluating V17 Replica Ranker Variants on {len(present_df)} present cases...")
    
    # We will test 4 ranking variants:
    # V14 Baseline: Standard CAR (correlation + conditional context + phase penalty)
    # V17-R1: CAR + Center Distance Prior (Gaussian penalty on distance from center)
    # V17-R2: CAR + Center Distance Prior + Family-level consensus
    # V17-R3: Adaptive Center-Weighted CAR based on Periodicity Index
    
    records = []
    
    top1_hits = {"V14_baseline": 0, "V17_R1_center": 0, "V17_R2_adaptive": 0, "V17_R3_family_center": 0}
    retrieval_survived = 0
    
    for idx, row in present_df.iterrows():
        pair_id = row["pair_id"]
        gt_x = float(row["gt_x"])
        gt_y = float(row["gt_y"])
        set_type = row.get("set_type", "SetA")
        
        ref_path = os.path.join("data/phase2_dev", row["reference_path"])
        search_path = os.path.join("data/phase2_dev", row["search_path"])
        
        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
        sh, sw = search_img.shape[:2]
        search_cx, search_cy = sw / 2.0, sh / 2.0
        
        # 1. Pose
        pose_res = perform_pose_fallback_search(ref_img, search_img)
        est_scale = float(pose_res["best_scale"])
        est_theta = float(pose_res["best_theta"])
        corr_plane = pose_res["corr_plane"]
        rotated_template = pose_res["best_template"]
        th, tw = rotated_template.shape[:2]
        
        # 2. V16 Akhilesh Extractor
        cands = extract_candidates_akhilesh(
            corr_plane, tw, th, ref_img=ref_img, search_img=search_img,
            est_scale=est_scale, est_theta=est_theta
        )
        
        # Check if GT is in candidates
        gt_in_pool = any(np.hypot(c["cx"] - gt_x, c["cy"] - gt_y) <= 5.0 for c in cands)
        if gt_in_pool:
            retrieval_survived += 1
            
        # Cluster families
        cands = cluster_replica_families(cands, est_scale)
        
        # Enrich candidate features
        for c in cands:
            px, py = c["peak_x"], c["peak_y"]
            cx, cy = c["cx"], c["cy"]
            
            y1, y2 = max(0, int(py)), min(sh, int(py + th))
            x1, x2 = max(0, int(px)), min(sw, int(px + tw))
            search_crop = search_img[y1:y2, x1:x2]
            
            psr, _, _ = compute_psr(corr_plane, px, py)
            ctx_res = verify_candidate_context(ref_img, search_img, cx, cy, est_scale, est_theta)
            phase_penalty = verify_phase_consistency(search_img, rotated_template, px, py)
            dist_to_center = np.hypot(cx - search_cx, cy - search_cy)
            
            c["psr"] = psr
            c["context_128"] = ctx_res.get("s128", 0.0)
            c["context_combined"] = ctx_res.get("combined", 0.0)
            c["phase_penalty"] = phase_penalty
            c["dist_to_center"] = dist_to_center
            
        # --- Variant 0: V14 Baseline CAR ---
        # Sort candidates using V14 ranking logic
        cands_v14 = list(cands)
        for c in cands_v14:
            # V14 formula
            c["score_v14"] = c["corr_score"] + 0.15 * c["context_128"] - 0.20 * c["phase_penalty"]
        cands_v14.sort(key=lambda x: x["score_v14"], reverse=True)
        if len(cands_v14) > 0 and np.hypot(cands_v14[0]["cx"] - gt_x, cands_v14[0]["cy"] - gt_y) <= 5.0:
            top1_hits["V14_baseline"] += 1
            
        # --- Variant 1: V17-R1 (CAR + Center Gaussian Prior) ---
        cands_r1 = list(cands)
        # Center prior sigma ~ 200 px (search image is 1024x1024, typical drift < 250 px)
        for c in cands_r1:
            center_penalty = (c["dist_to_center"] / 300.0) ** 2
            c["score_r1"] = c["corr_score"] + 0.15 * c["context_128"] - 0.20 * c["phase_penalty"] - 0.08 * center_penalty
        cands_r1.sort(key=lambda x: x["score_r1"], reverse=True)
        if len(cands_r1) > 0 and np.hypot(cands_r1[0]["cx"] - gt_x, cands_r1[0]["cy"] - gt_y) <= 5.0:
            top1_hits["V17_R1_center"] += 1
            
        # --- Variant 2: V17-R2 (Adaptive Center Weight based on Periodicity / Family Population) ---
        cands_r2 = list(cands)
        for c in cands_r2:
            # If candidate belongs to a large family (high periodic ambiguity), increase center prior
            fam_pop = c.get("family_population", 1)
            center_weight = 0.12 if fam_pop > 3 else 0.04
            center_penalty = (c["dist_to_center"] / 250.0) ** 2
            c["score_r2"] = c["corr_score"] + 0.15 * c["context_combined"] - 0.20 * c["phase_penalty"] - center_weight * center_penalty
        cands_r2.sort(key=lambda x: x["score_r2"], reverse=True)
        if len(cands_r2) > 0 and np.hypot(cands_r2[0]["cx"] - gt_x, cands_r2[0]["cy"] - gt_y) <= 5.0:
            top1_hits["V17_R2_adaptive"] += 1
            
        # --- Variant 3: V17-R3 (Combined Family Consensus + Center Distance Prior) ---
        cands_r3 = list(cands)
        for c in cands_r3:
            # Center Gaussian normalized
            center_prior = np.exp(-0.5 * (c["dist_to_center"] / 180.0) ** 2)
            c["score_r3"] = 0.40 * c["corr_score"] + 0.25 * c["context_combined"] - 0.15 * c["phase_penalty"] + 0.20 * center_prior
        cands_r3.sort(key=lambda x: x["score_r3"], reverse=True)
        if len(cands_r3) > 0 and np.hypot(cands_r3[0]["cx"] - gt_x, cands_r3[0]["cy"] - gt_y) <= 5.0:
            top1_hits["V17_R3_family_center"] += 1
            
        if (idx + 1) % 10 == 0 or idx == len(present_df) - 1:
            print(f"[{idx+1}/{len(present_df)}] Top-1 Hits: V14={top1_hits['V14_baseline']}, R1={top1_hits['V17_R1_center']}, R2={top1_hits['V17_R2_adaptive']}, R3={top1_hits['V17_R3_family_center']} (Pool Survived: {retrieval_survived})")
            
    total_present = len(present_df)
    print("\n" + "="*60)
    print(f"FINAL V17 REPLICA RANKER OFFLINE RESULTS ({total_present} PRESENT CASES):")
    print(f"GT Survived in Top-50 Pool: {retrieval_survived} / {total_present} ({retrieval_survived/total_present*100:.2f}%)")
    print("-" * 60)
    for model_name, hits in top1_hits.items():
        cond_acc = (hits / retrieval_survived) * 100 if retrieval_survived > 0 else 0.0
        abs_acc = (hits / total_present) * 100
        print(f"{model_name:25s}: Absolute Top-1 = {hits}/{total_present} ({abs_acc:.2f}%) | Conditional Top-1 = {hits}/{retrieval_survived} ({cond_acc:.2f}%)")
    print("="*60)

if __name__ == "__main__":
    evaluate_ranker_variants()
