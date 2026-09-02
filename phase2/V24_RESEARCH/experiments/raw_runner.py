import os
import sys
import numpy as np
import cv2

# Import Fallbacks
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from fallbacks.pose_fallback import perform_pose_fallback_search
from fallbacks.ranking_fallback import extract_candidates_fallback, rank_candidates_fallback
from fallbacks.rejection_fallback import evaluate_rejection_fallback

# Production modules import
sys.path.append("phase2")
from pose_refinement import refine_pose
from inference_phase2 import (
    verify_candidate_context,
    verify_phase_consistency,
    compute_psr,
    estimator_a_phase_correlation
)

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "production_engine"))
from config import POSE_ENGINE, RANKING_ENGINE, REJECTION_ENGINE

def run_production_localization(ref_img: np.ndarray, search_img: np.ndarray, 
                                verbose: bool = False) -> dict:
    """
    Master Production Integration Pipeline (Aashish Master System Owner):
    Orchestrates the entire Phase 2 pipeline.
    Uses Teammate modules as optional plug-ins under an explicit CONFIG selector,
    falling back to Aashish's proven robust implementations by default.
    """
    sh, sw = search_img.shape[:2]
    search_cx, search_cy = sw / 2.0, sh / 2.0
    
    # 1. Pose Estimation (Sai Plug-in or Aashish Fallback)
    pose_res = None
    if POSE_ENGINE != "fallback":
        try:
            sys.path.append("team/sai-pose")
            from pose_estimator import estimate_pose_sai
            pose_res = estimate_pose_sai(ref_img, search_img)
            if verbose: print(f"[Production Engine] Invoked specialist pose engine: {POSE_ENGINE}")
        except Exception as e:
            if verbose: print(f"[Production Engine] Warning: Failed to run Sai pose ({POSE_ENGINE}): {e}. Reverting to fallback.")
            
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
    if RANKING_ENGINE == "akhilesh_v1" and os.path.exists("team/akhilesh-localization/candidate_extractor.py"):
        try:
            sys.path.append("team/akhilesh-localization")
            from candidate_extractor import extract_candidates_akhilesh
            candidates = extract_candidates_akhilesh(corr_plane, tw, th, ref_img, search_img, est_scale, est_theta)
            if verbose: print("[Production Engine] Invoked specialist candidate extractor (Akhilesh).")
        except Exception as e:
            if verbose: print(f"[Production Engine] Specialist extractor failed: {e}. Reverting to fallback.")
            
    if candidates is None:
        candidates = extract_candidates_fallback(corr_plane, tw, th)
        
    # Enrich candidates with structural, context, and phase features (Aashish Baseline)
    enriched_candidates = []
    for c in candidates:
        px, py = c["peak_x"], c["peak_y"]
        cx, cy = c["cx"], c["cy"]
        
        y1, y2 = max(0, int(py)), min(sh, int(py + th))
        x1, x2 = max(0, int(px)), min(sw, int(px + tw))
        search_crop = search_img[y1:y2, x1:x2]
        
        # Compute PSR
        psr, _, _ = compute_psr(corr_plane, px, py)
        
        # Compute Context score
        context_res = verify_candidate_context(ref_img, search_img, cx, cy, est_scale, est_theta)
        
        # Compute Phase residual
        phase_dx, phase_dy, phase_residual = 0.0, 0.0, 0.0
        if search_crop.shape == (th, tw):
            phase_dx, phase_dy, phase_residual = estimator_a_phase_correlation(rotated_template, search_crop)
            
        # Compute Template residual (SSD)
        ssd = 0.0
        if search_crop.shape == (th, tw):
            ssd = float(np.mean((search_crop.astype(np.float32) - rotated_template.astype(np.float32)) ** 2))
            
        # Compute Phase consistency penalty
        phase_penalty = verify_phase_consistency(search_img, rotated_template, px, py)
        dist_to_center = np.hypot(cx - search_cx, cy - search_cy)
        
        enriched_candidates.append({
            "peak_x": px,
            "peak_y": py,
            "cx": cx,
            "cy": cy,
            "corr_score": c["corr_score"],
            "fft_gradient_score": 0.0,
            "peak_margin": 0.0,
            "psr": psr,
            "context_32": context_res["s32"],
            "context_64": context_res["s64"],
            "context_128": context_res["s128"],
            "context_score": context_res["combined"],
            "phase_dx": phase_dx,
            "phase_dy": phase_dy,
            "phase_residual": phase_residual,
            "phase_penalty": phase_penalty,
            "template_residual": ssd,
            "center_prior": dist_to_center,
            "score_combined": float(0.50 * c["corr_score"] + 0.50 * context_res["combined"] - phase_penalty),
            "pace_score": 0.0
        })
        
    # 3. Candidate Ranking (Akhilesh Plug-in or Aashish Fallback)
    ranked_candidates = None
    if RANKING_ENGINE != "fallback":
        try:
            sys.path.append("team/akhilesh-localization")
            from replica_ranker import rank_candidates_akhilesh
            ranked_candidates = rank_candidates_akhilesh(enriched_candidates, ref_img, search_img, est_scale, est_theta)
            if verbose: print(f"[Production Engine] Invoked specialist ranking engine: {RANKING_ENGINE}")
        except Exception as e:
            if verbose: print(f"[Production Engine] Specialist ranker ({RANKING_ENGINE}) failed: {e}. Reverting to fallback.")
            
    if ranked_candidates is None:
        ranked_candidates = rank_candidates_fallback(enriched_candidates, ref_img, search_img, est_scale, est_theta)
        
    # 4. Metrology & Subpixel Refinement (Always Run)
    if len(ranked_candidates) > 0:
        best_candidate = ranked_candidates[0]
        rx, ry, _, _ = refine_pose(ref_img, search_img, est_scale, est_theta, best_candidate["peak_x"], best_candidate["peak_y"], corr_plane)
        best_candidate["cx"] = rx
        best_candidate["cy"] = ry
    else:
        best_candidate = None
        
    # 5. Presence Rejection & Confidence Calibration (Shanganidhi Plug-in or Aashish Fallback)
    found = 1
    confidence = 0.0
    if REJECTION_ENGINE != "fallback":
        try:
            sys.path.append("team/shanganidhi-rejection")
            from rejection_model import evaluate_rejection_shanganidhi
            found, confidence = evaluate_rejection_shanganidhi(best_candidate)
            if verbose: print(f"[Production Engine] Invoked specialist rejection engine: {REJECTION_ENGINE}")
        except Exception as e:
            if verbose: print(f"[Production Engine] Specialist rejection ({REJECTION_ENGINE}) failed: {e}. Reverting to fallback.")
            
    if confidence == 0.0:
        found, confidence = evaluate_rejection_fallback(best_candidate, corr_plane, rotated_template, search_img)
        
    # Format Phase 2 required outputs
    x_out = best_candidate["cx"] if best_candidate is not None else 0.0
    y_out = best_candidate["cy"] if best_candidate is not None else 0.0
    theta_out = est_theta
    scale_out = est_scale
    
    return {
        "x": float(x_out),
        "y": float(y_out),
        "theta": float(theta_out),
        "scale": float(scale_out),
        "found": int(found),
        "score": float(confidence)
    }
