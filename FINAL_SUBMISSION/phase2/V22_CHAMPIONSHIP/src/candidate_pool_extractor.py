import os
import sys
import numpy as np
import pandas as pd
import cv2

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_dir)
sys.path.append("team/sai-pose")
sys.path.append("team/akhilesh-localization")
sys.path.append("phase2")

from fallbacks.pose_fallback import perform_pose_fallback_search
from candidate_extractor import extract_candidates_akhilesh
from inference_phase2 import (
    verify_candidate_context,
    verify_phase_consistency,
    compute_psr,
    estimator_a_phase_correlation
)

def build_candidate_pools(pairs_csv, output_csv):
    df_pairs = pd.read_csv(pairs_csv)
    # Only process pairs that are PRESENT (if you want all PRESENT pairs? The prompt says "For each pair in the dev set... GT info: data/phase2_dev/pairs.csv columns gt_x, gt_y, gt_found, set_type"). Wait, it also says "140 PRESENT pairs in data/phase2_dev/pairs.csv". I should probably only use PRESENT pairs or all pairs and filter later. Let's filter to gt_found == 1.
    df_pairs = df_pairs[df_pairs['gt_found'] == 1].reset_index(drop=True)
    
    all_features = []
    
    for idx, row in df_pairs.iterrows():
        pair_id = row['pair_id']
        ref_path = row['reference_path']
        search_path = row['search_path']
        gt_x = row['gt_x']
        gt_y = row['gt_y']
        
        print(f"Processing pair {idx}/{len(df_pairs)}: {pair_id}")
        
        ref_path_full = os.path.join("data/phase2_dev", ref_path)
        search_path_full = os.path.join("data/phase2_dev", search_path)
        
        ref_img = cv2.imread(ref_path_full, cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(search_path_full, cv2.IMREAD_GRAYSCALE)
        
        sh, sw = search_img.shape[:2]
        search_cx, search_cy = sw / 2.0, sh / 2.0
        
        # Pose Estimation
        pose_res = perform_pose_fallback_search(ref_img, search_img)
        est_scale = pose_res["best_scale"]
        est_theta = pose_res["best_theta"]
        corr_plane = pose_res["corr_plane"]
        rotated_template = pose_res["best_template"]
        th, tw = rotated_template.shape[:2]
        
        # Candidate Extraction (GET ALL 200)
        # The prompt says: run pose estimation + V19 candidate extraction to get ALL candidates (not just top-1)
        # extract_candidates_akhilesh takes max_final_k. We can pass max_final_k=200 to get all of them.
        candidates = extract_candidates_akhilesh(corr_plane, tw, th, ref_img, search_img, est_scale, est_theta, max_final_k=200)
        
        enriched_candidates = []
        corr_scores = []
        dist_to_centers = []
        phase_residuals = []
        context_128s = []
        
        for c in candidates:
            px, py = c["peak_x"], c["peak_y"]
            cx, cy = c["cx"], c["cy"]
            
            y1, y2 = max(0, int(py)), min(sh, int(py + th))
            x1, x2 = max(0, int(px)), min(sw, int(px + tw))
            search_crop = search_img[y1:y2, x1:x2]
            
            psr, _, _ = compute_psr(corr_plane, px, py)
            
            context_res = verify_candidate_context(ref_img, search_img, cx, cy, est_scale, est_theta)
            
            phase_dx, phase_dy, phase_residual = 0.0, 0.0, 0.0
            if search_crop.shape == (th, tw):
                phase_dx, phase_dy, phase_residual = estimator_a_phase_correlation(rotated_template, search_crop)
                
            phase_penalty = verify_phase_consistency(search_img, rotated_template, px, py)
            dist_to_center = np.hypot(cx - search_cx, cy - search_cy)
            
            # Distance from GT
            err_x = cx - gt_x
            err_y = cy - gt_y
            dist_to_gt = np.hypot(err_x, err_y)
            tolerance = max(25.0, tw * 0.25)
            is_correct = 1 if dist_to_gt <= tolerance else 0
            
            c_feat = {
                "pair_id": pair_id,
                "cx": cx,
                "cy": cy,
                "corr_score": c["corr_score"],
                "psr": psr,
                "context_128": context_res["s128"],
                "phase_residual": phase_residual,
                "phase_penalty": phase_penalty,
                "dist_to_center": dist_to_center,
                "peak_margin": c.get("corr_score", 0), # peak_margin? usually same as corr_score for now?
                "is_correct": is_correct
            }
            
            enriched_candidates.append(c_feat)
            corr_scores.append(c["corr_score"])
            dist_to_centers.append(dist_to_center)
            phase_residuals.append(phase_residual)
            context_128s.append(context_res["s128"])
            
        # Compute Relatives
        med_corr = np.median(corr_scores)
        med_dist = np.median(dist_to_centers)
        med_phase = np.median(phase_residuals)
        med_context = np.median(context_128s)
        
        # Rank features
        ranks_ncc = np.argsort(np.argsort([-c['corr_score'] for c in enriched_candidates])) + 1
        ranks_phase = np.argsort(np.argsort([c['phase_residual'] for c in enriched_candidates])) + 1
        ranks_center = np.argsort(np.argsort([c['dist_to_center'] for c in enriched_candidates])) + 1
        
        # Family grouping (simple spatial grouping for family size)
        # Two candidates are in the same family if they are within max(tw, th) of each other
        family_radius = max(tw, th)
        for i, c in enumerate(enriched_candidates):
            c['ncc_delta'] = c['corr_score'] - med_corr
            c['center_delta'] = c['dist_to_center'] - med_dist
            c['phase_delta'] = c['phase_residual'] - med_phase
            c['context_delta'] = c['context_128'] - med_context
            c['rank_by_ncc'] = ranks_ncc[i]
            c['rank_by_phase'] = ranks_phase[i]
            c['rank_by_center'] = ranks_center[i]
            
            # compute family pop
            family_members = [
                oc for j, oc in enumerate(enriched_candidates)
                if np.hypot(oc['cx'] - c['cx'], oc['cy'] - c['cy']) <= family_radius
            ]
            c['family_size'] = len(family_members)
            c['family_population'] = len(family_members)
            
            if len(family_members) > 1:
                # find 2nd best in family
                family_nccs = sorted([fm['corr_score'] for fm in family_members], reverse=True)
                c['nearest_competitor_ncc'] = family_nccs[1] if family_nccs[0] == c['corr_score'] else family_nccs[0]
            else:
                c['nearest_competitor_ncc'] = 0.0
                
        all_features.extend(enriched_candidates)
        
    df_out = pd.DataFrame(all_features)
    df_out.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")

if __name__ == "__main__":
    os.makedirs("phase2/V22_CHAMPIONSHIP/results", exist_ok=True)
    build_candidate_pools("data/phase2_dev/pairs.csv", "phase2/V22_CHAMPIONSHIP/results/candidate_pool_features.csv")
