import time
import cv2
import numpy as np
import os
import sys

from scale_search import coarse_to_fine_scale_search
from rotation_search import coarse_to_fine_rotation_search
from rejection import extract_presence_features, classify_presence, compute_psr
from calibration import calibrate_confidence_score
from pose_refinement import refine_pose, estimator_a_phase_correlation
from context_matcher import verify_candidate_context
from phase_verifier import verify_phase_consistency
from periodicity_detector import compute_ambiguity_index
from conditional_pace import rerank_with_pace
from channel_consensus import extract_gradient

# V10 Modules
from family_clustering import cluster_replica_families
from spatial_fingerprint import compute_spatial_fingerprint
from candidate_ranker import rank_candidates, log_candidate_features
from adaptive_peak_detector import extract_adaptive_candidates

def compute_rgb_similarity(ref_bgr: np.ndarray, search_bgr: np.ndarray, 
                           px: int, py: int, scale: float, theta: float) -> float:
    """
    Computes color similarity between reference BGR and search candidate crop
    using HSV color histogram intersection. Returns score in [0.0, 1.0].
    """
    try:
        ref_h, ref_w = ref_bgr.shape[:2]
        tw = int(round(ref_w / scale))
        th = int(round(ref_h / scale))
        sh, sw = search_bgr.shape[:2]
        
        y1, y2 = max(0, int(py)), min(sh, int(py + th))
        x1, x2 = max(0, int(px)), min(sw, int(px + tw))
        search_crop = search_bgr[y1:y2, x1:x2]
        
        if search_crop.shape[0] < th // 2 or search_crop.shape[1] < tw // 2:
            return 0.0
            
        # Resize reference to match candidate scale
        ref_resized = cv2.resize(ref_bgr, (search_crop.shape[1], search_crop.shape[0]), interpolation=cv2.INTER_AREA)
        
        # Convert to HSV
        hsv_ref = cv2.cvtColor(ref_resized, cv2.COLOR_BGR2HSV)
        hsv_search = cv2.cvtColor(search_crop, cv2.COLOR_BGR2HSV)
        
        # Compute histograms
        hist_ref = cv2.calcHist([hsv_ref], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hist_search = cv2.calcHist([hsv_search], [0, 1], None, [50, 60], [0, 180, 0, 256])
        
        cv2.normalize(hist_ref, hist_ref, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_search, hist_search, 0, 1, cv2.NORM_MINMAX)
        
        similarity = cv2.compareHist(hist_ref, hist_search, cv2.HISTCMP_CORREL)
        return float(max(0.0, similarity))
    except Exception:
        return 0.0

def perform_phase2_localization(ref_input, search_input, gt_x=None, gt_y=None, gt_found=None, pair_id=None) -> dict:
    """
    Drift-Sense++ SAFE-CAR 2 (V10 Final Engine):
    1. Scale Search.
    2. Rotation Search.
    3. Peak Suppression (Top-20 Candidates).
    4. Replica Family Clustering.
    5. Spatial Fingerprint Matching.
    6. Candidate Evidence Ranking & Logging.
    7. Ambiguity-gated PACE Re-ranking.
    8. Rejection & Calibration.
    9. Set D RGB Bonus (Hist intersection).
    """
    t_start = time.perf_counter()

    # Load BGR copies for color similarity checks
    ref_bgr, search_bgr = None, None
    if isinstance(ref_input, str):
        ref_raw = cv2.imread(ref_input)
        if ref_raw is not None and len(ref_raw.shape) == 3:
            ref_bgr = ref_raw
            ref_img = cv2.cvtColor(ref_raw, cv2.COLOR_BGR2GRAY)
        else:
            ref_img = cv2.imread(ref_input, cv2.IMREAD_GRAYSCALE)
    else:
        ref_raw = ref_input.copy()
        if len(ref_raw.shape) == 3:
            ref_bgr = ref_raw
            ref_img = cv2.cvtColor(ref_raw, cv2.COLOR_BGR2GRAY)
        else:
            ref_img = ref_raw

    if isinstance(search_input, str):
        search_raw = cv2.imread(search_input)
        if search_raw is not None and len(search_raw.shape) == 3:
            search_bgr = search_raw
            search_img = cv2.cvtColor(search_raw, cv2.COLOR_BGR2GRAY)
        else:
            search_img = cv2.imread(search_input, cv2.IMREAD_GRAYSCALE)
    else:
        search_raw = search_input.copy()
        if len(search_raw.shape) == 3:
            search_bgr = search_raw
            search_img = cv2.cvtColor(search_raw, cv2.COLOR_BGR2GRAY)
        else:
            search_img = search_raw

    if ref_img is None or search_img is None:
        raise ValueError("Failed to load reference or search image.")

    sh, sw = search_img.shape[:2]
    search_cx, search_cy = sw / 2.0, sh / 2.0

    # Pre-extract gradients
    search_grad = extract_gradient(search_img)

    # 1. Scale Search
    scale_res = coarse_to_fine_scale_search(ref_img, search_img, scale_min=8.0, scale_max=12.0)
    est_scale = scale_res["best_scale"]
    best_template = scale_res["best_template"]

    # 2. Rotation Search
    rot_res = coarse_to_fine_rotation_search(best_template, search_img, angle_min=-5.0, angle_max=5.0)
    est_theta = rot_res["best_theta"]
    corr_plane = rot_res["corr_plane"]
    rotated_template = rot_res["rotated_template"]
    
    rotated_template_grad = extract_gradient(rotated_template)

    # 3. Adaptive Peak Detection (V11.1)
    th, tw = rotated_template.shape[:2]
    corr_grad = cv2.matchTemplate(search_grad, rotated_template_grad, cv2.TM_CCOEFF_NORMED)
    
    # Extract up to 50 candidates adaptively
    raw_candidates = extract_adaptive_candidates(corr_plane, corr_grad, tw, th, max_k=50)
    
    # Apply Adaptive K (Dynamic Truncation for speed)
    if len(raw_candidates) > 1:
        peak_margin = raw_candidates[0]["corr_score"] - raw_candidates[1]["corr_score"]
        # If very clean, isolated match, truncate K to 10 candidates to keep latency low
        if peak_margin >= 0.15 and raw_candidates[0]["corr_score"] > 0.80:
            raw_candidates = raw_candidates[:10]
            
    # Enrich candidates with structural, context, and phase features
    candidates = []
    for c in raw_candidates:
        px, py = c["peak_x"], c["peak_y"]
        cx, cy = c["cx"], c["cy"]
        
        y1, y2 = max(0, int(py)), min(sh, int(py + th))
        x1, x2 = max(0, int(px)), min(sw, int(px + tw))
        search_crop = search_img[y1:y2, x1:x2]
        
        psr, _, _ = compute_psr(corr_plane, px, py)
        
        g_val = 0.0
        if py < corr_grad.shape[0] and px < corr_grad.shape[1]:
            g_val = float(corr_grad[py, px])
            
        context_res = verify_candidate_context(ref_img, search_img, cx, cy, est_scale, est_theta)
        
        phase_dx, phase_dy, phase_residual = 0.0, 0.0, 0.0
        if search_crop.shape == (th, tw):
            phase_dx, phase_dy, phase_residual = estimator_a_phase_correlation(rotated_template, search_crop)
            
        ssd = 0.0
        if search_crop.shape == (th, tw):
            ssd = float(np.mean((search_crop.astype(np.float32) - rotated_template.astype(np.float32)) ** 2))
            
        phase_penalty = verify_phase_consistency(search_img, rotated_template, px, py)
        dist_to_center = np.hypot(cx - search_cx, cy - search_cy)
        
        candidates.append({
            "peak_x": px,
            "peak_y": py,
            "cx": cx,
            "cy": cy,
            "corr_score": c["corr_score"],
            "fft_gradient_score": g_val,
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
        
    for i in range(len(candidates)):
        next_score = candidates[i+1]["corr_score"] if i+1 < len(candidates) else 0.0
        candidates[i]["peak_margin"] = candidates[i]["corr_score"] - next_score

    # 4. Replica Family Clustering (V10.2)
    candidates = cluster_replica_families(candidates, est_scale)

    # 5. Spatial Fingerprint Matching (V10.3)
    for c in candidates:
        fam_members = [m for m in candidates if m.get("family_id") == c.get("family_id")]
        fp = compute_spatial_fingerprint(search_img, c["cx"], c["cy"], est_scale, fam_members)
        c.update(fp)

    # 6. Candidate Evidence Ranking & Logging (V10.4)
    candidates = rank_candidates(candidates)
    if pair_id is not None and gt_x is not None:
        log_candidate_features(pair_id, candidates, gt_x, gt_y, gt_found)

    # 7. Periodicity Lattice Detection (Ambiguity Index)
    ambiguity_score, is_ambiguous = compute_ambiguity_index(candidates, est_scale)
    for c in candidates:
        c["periodicity_index"] = ambiguity_score
        
    # 8. Conditional PACE Context Re-Ranking
    if is_ambiguous and len(candidates) > 0:
        candidates = rerank_with_pace(ref_img, search_img, candidates, est_scale, est_theta)
        
        # Apply Center-Prior Tie-Breaker
        for cand in candidates:
            center_penalty = float(0.08 * (cand["center_prior"] / (sw / 2.0)))
            cand["rank_score"] = cand["rank_score"] - center_penalty
            
        # Re-sort after PACE and center-prior
        candidates.sort(key=lambda x: x.get("rank_score", 0.0), reverse=True)

    # Select best candidate after final ranking
    if len(candidates) == 0:
        _, _, _, max_loc = cv2.minMaxLoc(corr_plane)
        peak_x, peak_y = max_loc
        best_cand = None
    else:
        best_cand = candidates[0]
        peak_x = best_cand["peak_x"]
        peak_y = best_cand["peak_y"]

    # 9. Presence Classification & Rejection (Set C)
    if best_cand is not None:
        context_score = best_cand["context_score"]
        phase_residual = best_cand["phase_residual"]
    else:
        context_score = 0.0
        phase_residual = 0.0

    presence_feats = extract_presence_features(
        corr_plane, peak_x, peak_y, rotated_template, search_img,
        context_score=context_score, phase_residual=phase_residual
    )
    found, raw_presence_score = classify_presence(presence_feats)

    # 10. Confidence Score Calibration
    if found == 0:
        decision_confidence = 1.0 - raw_presence_score
    else:
        decision_confidence = raw_presence_score
        
    calibrated_score = calibrate_confidence_score(decision_confidence)

    # 11. RGB Optical Channel Support (Set D Bonus)
    if found == 1 and ref_bgr is not None and search_bgr is not None:
        rgb_sim = compute_rgb_similarity(ref_bgr, search_bgr, peak_x, peak_y, est_scale, est_theta)
        if rgb_sim > 0.85:
            calibrated_score = min(1.0, calibrated_score + 0.05)

    # 12. Output Refinement
    if found == 0:
        x, y = 0.0, 0.0
        final_theta = 0.0
        final_scale = 0.0
    else:
        x, y, final_scale, final_theta = refine_pose(
            ref_img, search_img, est_scale, est_theta, peak_x, peak_y, corr_plane
        )

    t_end = time.perf_counter()
    latency_ms = (t_end - t_start) * 1000.0

    return {
        "x": float(np.round(x, 2)),
        "y": float(np.round(y, 2)),
        "theta": float(np.round(final_theta, 4)),
        "scale": float(np.round(final_scale, 4)),
        "found": int(found),
        "score": float(np.round(calibrated_score, 4)),
        "latency_ms": float(np.round(latency_ms, 2))
    }
