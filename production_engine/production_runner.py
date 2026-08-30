import os
import sys
import numpy as np
import cv2

# Import Fallbacks
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fallbacks.pose_fallback import perform_pose_fallback_search
from fallbacks.ranking_fallback import extract_candidates_fallback, rank_candidates_fallback
from fallbacks.rejection_fallback import evaluate_rejection_fallback

# Production modules import
sys.path.append("phase2")
from pose_refinement import refine_pose
from inference_phase2 import verify_candidate_context, verify_phase_consistency, compute_psr

def run_production_localization(ref_img: np.ndarray, search_img: np.ndarray, 
                                verbose: bool = False) -> dict:
    """
    Master Production Integration Pipeline (Aashish Master System Owner):
    Orchestrates the entire Phase 2 pipeline.
    Uses Teammate modules as optional plug-ins, falling back to Aashish's
    proven robust implementations if teammate files are missing or unvalidated.
    """
    sh, sw = search_img.shape[:2]
    search_cx, search_cy = sw / 2.0, sh / 2.0
    
    # 1. Pose Estimation (Sai Plug-in or Aashish Fallback)
    sai_pose_path = "team/sai-pose/pose_estimator.py"
    pose_res = None
    if os.path.exists(sai_pose_path):
        try:
            sys.path.append("team/sai-pose")
            from pose_estimator import estimate_pose_sai
            pose_res = estimate_pose_sai(ref_img, search_img)
            if verbose: print("[Production Engine] Successfully invoked Sai's pose estimator.")
        except Exception as e:
            if verbose: print(f"[Production Engine] Warning: Failed to run Sai's pose: {e}. Switching to Aashish fallback.")
            
    if pose_res is None:
        pose_res = perform_pose_fallback_search(ref_img, search_img)
        if verbose: print("[Production Engine] Ran Aashish fallback pose estimator.")
        
    est_scale = pose_res["best_scale"]
    est_theta = pose_res["best_theta"]
    corr_plane = pose_res["corr_plane"]
    rotated_template = pose_res["best_template"]
    th, tw = rotated_template.shape[:2]
    
    # 2. Candidate Proposals (Sai/Akhilesh Plug-in or Aashish Fallback)
    candidates = None
    if os.path.exists("team/sai-pose/candidate_extractor.py"):
        try:
            sys.path.append("team/sai-pose")
            from candidate_extractor import extract_candidates_sai
            candidates = extract_candidates_sai(corr_plane, tw, th)
        except Exception as e:
            if verbose: print(f"[Production Engine] Teammate extractor failed: {e}.")
            
    if candidates is None:
        candidates = extract_candidates_fallback(corr_plane, tw, th)
        
    # Enrich candidates with structural, context, and phase features (Aashish Baseline)
    for c in candidates:
        px, py = c["peak_x"], c["peak_y"]
        cx, cy = c["cx"], c["cy"]
        
        # Compute PSR
        psr, _, _ = compute_psr(corr_plane, px, py)
        c["psr"] = psr
        
        # Compute Context score
        context_res = verify_candidate_context(ref_img, search_img, cx, cy, est_scale, est_theta)
        c["context_score"] = context_res["combined"]
        
        # Compute Phase consistency penalty
        phase_penalty = verify_phase_consistency(search_img, rotated_template, px, py)
        c["phase_penalty"] = phase_penalty
        
    # 3. Candidate Ranking (Akhilesh Plug-in or Aashish Fallback)
    ranked_candidates = None
    if os.path.exists("team/akhilesh-localization/replica_ranker.py"):
        try:
            sys.path.append("team/akhilesh-localization")
            from replica_ranker import rank_candidates_akhilesh
            ranked_candidates = rank_candidates_akhilesh(candidates)
        except Exception as e:
            if verbose: print(f"[Production Engine] Teammate ranker failed: {e}.")
            
    if ranked_candidates is None:
        ranked_candidates = rank_candidates_fallback(candidates)
        
    # 4. Metrology & Subpixel Refinement (Always Run)
    if len(ranked_candidates) > 0:
        best = ranked_candidates[0]
        rx, ry, _, _ = refine_pose(ref_img, search_img, est_scale, est_theta, best["peak_x"], best["peak_y"], corr_plane)
        best_candidate = best
        best_candidate["cx"] = rx
        best_candidate["cy"] = ry
    else:
        best_candidate = None
        
    # 5. Presence Rejection & Confidence Calibration (Shanganidhi Plug-in or Aashish Fallback)
    found = 1
    confidence = 0.0
    if os.path.exists("team/shanganidhi-rejection/rejection_model.py"):
        try:
            sys.path.append("team/shanganidhi-rejection")
            from rejection_model import evaluate_rejection_shanganidhi
            found, confidence = evaluate_rejection_shanganidhi(best_candidate)
        except Exception as e:
            if verbose: print(f"[Production Engine] Teammate rejection failed: {e}.")
            
    if confidence == 0.0:
        found, confidence = evaluate_rejection_fallback(best_candidate, is_ambiguous=False)
        
    # Format Phase 2 required outputs
    x_out = best_candidate["cx"] if (best_candidate is not None and found == 1) else 0.0
    y_out = best_candidate["cy"] if (best_candidate is not None and found == 1) else 0.0
    theta_out = est_theta if found == 1 else 0.0
    scale_out = est_scale if found == 1 else 0.0
    
    return {
        "x": float(x_out),
        "y": float(y_out),
        "theta": float(theta_out),
        "scale": float(scale_out),
        "found": int(found),
        "score": float(confidence)
    }
