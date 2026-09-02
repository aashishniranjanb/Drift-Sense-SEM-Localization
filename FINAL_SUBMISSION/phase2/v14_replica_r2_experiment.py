import os
import sys
import time
import cv2
import numpy as np
import pandas as pd

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, "phase2"))
sys.path.append(os.path.join(parent_dir, "fallbacks"))
sys.path.append(os.path.join(parent_dir, "production_engine"))

from scale_search import coarse_to_fine_scale_search
from rotation_search import coarse_to_fine_rotation_search
from pose_refinement import refine_pose
from fallbacks.ranking_fallback import extract_candidates_fallback
from inference_phase2 import (
    verify_candidate_context,
    verify_phase_consistency,
    compute_psr,
    estimator_a_phase_correlation,
    cluster_replica_families,
    compute_spatial_fingerprint,
    rank_candidates,
    compute_ambiguity_index,
    rerank_with_pace
)

def evaluate_v14_r2():
    pairs_df = pd.read_csv("data/phase2_dev/pairs.csv")
    
    # -------------------------------------------------------------
    # Run V14-R2 (Enhanced Context-128 & Family Consistency Ranker)
    # -------------------------------------------------------------
    print("Evaluating V14-R2 Ranking formulation on all 180 dev pairs...")
    results_r2 = []
    start_time = time.time()
    
    for idx, row in pairs_df.iterrows():
        pair_id = row["pair_id"]
        gt_found = int(row["gt_found"])
        set_type = row.get("set_type", "SetA" if gt_found == 1 else "SetC")
        gt_x = float(row["gt_x"])
        gt_y = float(row["gt_y"])
        
        ref_img = cv2.imread(os.path.join("data/phase2_dev", row["reference_path"]), cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(os.path.join("data/phase2_dev", row["search_path"]), cv2.IMREAD_GRAYSCALE)
        sh, sw = search_img.shape[:2]
        
        # 1. Pose estimation (Sequential Fallback)
        scale_res = coarse_to_fine_scale_search(ref_img, search_img)
        rot_res = coarse_to_fine_rotation_search(scale_res["best_template"], search_img)
        
        est_scale = float(scale_res["best_scale"])
        est_theta = float(rot_res["best_theta"])
        rotated_tpl = rot_res["rotated_template"]
        th, tw = rotated_tpl.shape[:2]
        corr_plane = rot_res["corr_plane"]
        
        # 2. Extract Top-50 candidates
        cands = extract_candidates_fallback(corr_plane, tw, th, max_k=50)
        
        # 3. Enrich candidates
        enriched = []
        for c in cands:
            px, py = c["peak_x"], c["peak_y"]
            cx, cy = c["cx"], c["cy"]
            y1, y2 = max(0, int(py)), min(sh, int(py + th))
            x1, x2 = max(0, int(px)), min(sw, int(px + tw))
            search_crop = search_img[y1:y2, x1:x2]
            
            psr, _, _ = compute_psr(corr_plane, px, py)
            ctx_res = verify_candidate_context(ref_img, search_img, cx, cy, est_scale, est_theta)
            
            phase_dx, phase_dy, phase_residual = 0.0, 0.0, 0.0
            if search_crop.shape == (th, tw):
                phase_dx, phase_dy, phase_residual = estimator_a_phase_correlation(rotated_tpl, search_crop)
            phase_penalty = verify_phase_consistency(search_img, rotated_tpl, px, py)
            dist_to_center = np.hypot(cx - sw/2.0, cy - sh/2.0)
            
            enriched.append({
                "peak_x": px, "peak_y": py, "cx": cx, "cy": cy,
                "corr_score": c["corr_score"], "psr": psr,
                "context_32": ctx_res["s32"], "context_64": ctx_res["s64"],
                "context_128": ctx_res["s128"], "context_score": ctx_res["combined"],
                "phase_dx": phase_dx, "phase_dy": phase_dy,
                "phase_residual": phase_residual, "phase_penalty": phase_penalty,
                "center_prior": dist_to_center,
                "score_combined": float(0.50 * c["corr_score"] + 0.50 * ctx_res["combined"] - phase_penalty)
            })
            
        for i in range(len(enriched)):
            next_score = enriched[i+1]["corr_score"] if i+1 < len(enriched) else 0.0
            enriched[i]["peak_margin"] = enriched[i]["corr_score"] - next_score

        # 4. Apply V14-R2 Ranking logic:
        # Step A: Cluster replica families
        if len(enriched) > 0:
            enriched = cluster_replica_families(enriched, est_scale)
            for c in enriched:
                fam_members = [m for m in enriched if m.get("family_id") == c.get("family_id")]
                fp = compute_spatial_fingerprint(search_img, c["cx"], c["cy"], est_scale, fam_members)
                c.update(fp)
            
            # Step B: Compute Ambiguity
            ambiguity_score, is_ambiguous = compute_ambiguity_index(enriched, est_scale)
            
            if is_ambiguous:
                # If ambiguous periodic lattice, use context_128 as the primary tie-breaker
                for cand in enriched:
                    # R2 composite score
                    cand["rank_score"] = float(
                        0.30 * cand["corr_score"] + 
                        0.45 * cand["context_128"] + 
                        0.15 * cand["context_64"] - 
                        0.15 * cand["phase_residual"] - 
                        0.05 * (cand["center_prior"] / (sw / 2.0))
                    )
                enriched.sort(key=lambda x: x.get("rank_score", 0.0), reverse=True)
            else:
                # If clean match, trust standard CAR combined score
                enriched.sort(key=lambda x: x.get("score_combined", 0.0), reverse=True)
                
            best_cand = enriched[0]
            rx, ry, _, _ = refine_pose(ref_img, search_img, est_scale, est_theta, best_cand["peak_x"], best_cand["peak_y"], corr_plane)
            best_cand["cx"] = rx
            best_cand["cy"] = ry
        else:
            best_cand = None

        # 5. Presence Evaluation (V14-P1 engine)
        if best_cand is not None:
            corr = best_cand["corr_score"]
            psr = best_cand["psr"]
            margin = best_cand["peak_margin"]
            ctx128 = best_cand["context_128"]
            phase_res = best_cand["phase_residual"]
            comp = float(0.35 * corr + 0.40 * ctx128 + 0.15 * (psr / 10.0) + 0.10 * margin - 0.20 * phase_res)
            calibrated_score = max(0.0, min(1.0, comp))
            found = 1 if calibrated_score >= 0.58 else 0
        else:
            calibrated_score = 0.0
            found = 0
            
        loc_err = -1.0
        if gt_found == 1 and found == 1 and best_cand is not None:
            loc_err = float(np.hypot(best_cand["cx"] - gt_x, best_cand["cy"] - gt_y))
            
        results_r2.append({
            "pair_id": pair_id,
            "set_type": set_type,
            "gt_found": gt_found,
            "found": found,
            "loc_err": loc_err,
            "score": calibrated_score
        })
        
    total_time = time.time() - start_time
    df_res = pd.DataFrame(results_r2)
    
    setA = df_res[df_res["set_type"] == "SetA"]
    setB = df_res[df_res["set_type"] == "SetB"]
    
    setA_le5 = np.mean((setA["found"] == 1) & (setA["loc_err"] >= 0) & (setA["loc_err"] <= 5.0)) * 100.0 if len(setA) > 0 else 0.0
    setB_le5 = np.mean((setB["found"] == 1) & (setB["loc_err"] >= 0) & (setB["loc_err"] <= 5.0)) * 100.0 if len(setB) > 0 else 0.0
    weighted_loc = 0.45 * setA_le5 + 0.55 * setB_le5
    
    # Rejection F1
    tp_rej = np.sum((df_res["gt_found"] == 0) & (df_res["found"] == 0))
    fp_rej = np.sum((df_res["gt_found"] == 1) & (df_res["found"] == 0))
    fn_rej = np.sum((df_res["gt_found"] == 0) & (df_res["found"] == 1))
    prec_rej = tp_rej / (tp_rej + fp_rej) if (tp_rej + fp_rej) > 0 else 0.0
    rec_rej = tp_rej / (tp_rej + fn_rej) if (tp_rej + fn_rej) > 0 else 0.0
    f1_rej = 2 * prec_rej * rec_rej / (prec_rej + rec_rej) if (prec_rej + rec_rej) > 0 else 0.0
    
    print("\n--- V14-R2 Experiment Results ---")
    print(f"Weighted Loc Score: {weighted_loc:.2f}% (Set A: {setA_le5:.2f}%, Set B: {setB_le5:.2f}%)")
    print(f"Set C Rejection F1: {f1_rej:.4f} (Precision: {prec_rej:.4f}, Recall: {rec_rej:.4f})")
    print(f"Total Runtime: {total_time:.2f}s ({total_time/len(pairs_df):.2f}s/pair)")
    
    # Comparison table
    comparison_md = f"""# V14-R2 Replica Ranking Experiment

| Metric | V14 Baseline | V14-R2 (Context-128 Ambiguity Filter) | Delta | Decision |
| :--- | :---: | :---: | :---: | :--- |
| **Weighted Loc ($\\le 5\\text{{ px}}$)** | **48.88%** | **{weighted_loc:.2f}%** | **{weighted_loc - 48.88:+.2f}%** | {'ADOPT' if weighted_loc > 48.88 else 'RETAIN V14 BASELINE'} |
| **Set A $\\le 5\\text{{ px}}$** | 38.78% | {setA_le5:.2f}% | {setA_le5 - 38.78:+.2f}% | — |
| **Set B $\\le 5\\text{{ px}}$** | 57.14% | {setB_le5:.2f}% | {setB_le5 - 57.14:+.2f}% | — |
| **Rejection F1** | 0.3862 | {f1_rej:.4f} | {f1_rej - 0.3862:+.4f} | — |
| **Avg Latency** | 2.95s | {total_time/len(pairs_df):.2f}s | — | Viable (<5s) |
"""
    with open("results/v14/V14_R2_COMPARISON.md", "w") as f:
        f.write(comparison_md)
    print("Saved comparison to results/v14/V14_R2_COMPARISON.md")

if __name__ == "__main__":
    evaluate_v14_r2()
