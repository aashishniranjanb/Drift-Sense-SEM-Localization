import numpy as np
import cv2
from scipy.ndimage import maximum_filter

# Import robust CAR ranking functions from phase2 baseline
import sys
sys.path.append("phase2")
from inference_phase2 import (
    cluster_replica_families,
    compute_spatial_fingerprint,
    rank_candidates,
    compute_ambiguity_index,
    rerank_with_pace
)

def extract_candidates_fallback(corr_plane: np.ndarray, tw: int, th: int, max_k: int = 50) -> list:
    """
    Aashish Candidate Extraction Fallback:
    Extracts candidates using iterative NMS r=5.
    """
    ch, cw = corr_plane.shape[:2]
    candidates_list = []
    
    work = corr_plane.copy()
    for _ in range(max_k):
        _, max_val, _, max_loc = cv2.minMaxLoc(work)
        if max_val <= 0.01 or np.isnan(max_val): break
        px, py = max_loc
        cx = px + tw / 2.0
        cy = py + th / 2.0
        
        candidates_list.append({
            "peak_x": px,
            "peak_y": py,
            "cx": cx,
            "cy": cy,
            "corr_score": float(max_val)
        })
        
        y1, y2 = max(0, py - 5), min(ch, py + 6)
        x1, x2 = max(0, px - 5), min(cw, px + 6)
        work[y1:y2, x1:x2] = -999.0
        
    return candidates_list

def rank_candidates_fallback(candidates: list, ref_img: np.ndarray, search_img: np.ndarray, 
                             est_scale: float, est_theta: float) -> list:
    """
    Aashish Ranking Fallback (Full CAR Pipeline):
    Invokes the complete replica clustering, spatial fingerprinting, and PACE re-ranking engines.
    """
    if len(candidates) == 0:
        return []
        
    sh, sw = search_img.shape[:2]
    
    # 1. Cluster Replica Families
    candidates = cluster_replica_families(candidates, est_scale)
    
    # 2. Spatial Fingerprint Matching
    for c in candidates:
        fam_members = [m for m in candidates if m.get("family_id") == c.get("family_id")]
        fp = compute_spatial_fingerprint(search_img, c["cx"], c["cy"], est_scale, fam_members)
        c.update(fp)
        
    # 3. Base Ranking Rules
    candidates = rank_candidates(candidates)
    
    # 4. Periodicity Lattice Detection
    ambiguity_score, is_ambiguous = compute_ambiguity_index(candidates, est_scale)
    for c in candidates:
        c["periodicity_index"] = ambiguity_score
        
    # 5. Conditional PACE Context Re-Ranking
    if is_ambiguous and len(candidates) > 0:
        candidates = rerank_with_pace(ref_img, search_img, candidates, est_scale, est_theta)
        
        # Apply Center-Prior Tie-Breaker
        for cand in candidates:
            center_penalty = float(0.08 * (cand["center_prior"] / (sw / 2.0)))
            cand["rank_score"] = cand.get("rank_score", 0.0) - center_penalty
            
        # Re-sort after PACE and center-prior
        candidates.sort(key=lambda x: x.get("rank_score", 0.0), reverse=True)
    else:
        # Sort by default rank score or combined score
        candidates.sort(key=lambda x: x.get("score_combined", 0.0), reverse=True)
        
    return candidates
