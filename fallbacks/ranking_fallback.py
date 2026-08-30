import numpy as np
import cv2
from scipy.ndimage import maximum_filter

def extract_candidates_fallback(corr_plane: np.ndarray, tw: int, th: int, max_k: int = 100) -> list:
    """
    Aashish Candidate Extraction Fallback:
    Finds local maxima in 7x7 windows and runs iterative NMS with r=5 to recover peaks.
    """
    ch, cw = corr_plane.shape[:2]
    candidates_list = []
    
    # 1. Local Maxima
    size_int = 7
    local_max = (maximum_filter(corr_plane, size=size_int) == corr_plane) & (corr_plane > 0.05)
    iy, ix = np.where(local_max)
    for x, y in zip(ix, iy):
        candidates_list.append({"px": int(x), "py": int(y), "score": float(corr_plane[y, x]), "source": "lmax"})
        
    # 2. NMS Peaks
    work = corr_plane.copy()
    for _ in range(max_k):
        _, max_val, _, max_loc = cv2.minMaxLoc(work)
        if max_val <= 0.05 or np.isnan(max_val): break
        px, py = max_loc
        candidates_list.append({"px": int(px), "py": int(py), "score": float(max_val), "source": "nms"})
        y1, y2 = max(0, py - 5), min(ch, py + 6)
        x1, x2 = max(0, px - 5), min(cw, px + 6)
        work[y1:y2, x1:x2] = -999.0
        
    # Sort and Deduplicate
    candidates_list.sort(key=lambda x: x["score"], reverse=True)
    unique = []
    for c in candidates_list:
        px, py = c["px"], c["py"]
        cx = px + tw / 2.0
        cy = py + th / 2.0
        if not any(np.hypot(cx - u["cx"], cy - u["cy"]) < 3.0 for u in unique):
            unique.append({
                "peak_x": px,
                "peak_y": py,
                "cx": cx,
                "cy": cy,
                "corr_score": c["score"],
                "source": c["source"]
            })
            if len(unique) >= max_k:
                break
    return unique

def rank_candidates_fallback(candidates: list) -> list:
    """
    Aashish Ranking Fallback (Confidence-Adaptive Ranking):
    Ranks candidates using a combined score of correlation and context matching.
    """
    if len(candidates) == 0:
        return []
        
    # Standard combined ranker score
    for c in candidates:
        c["score_combined"] = float(0.50 * c["corr_score"] + 0.50 * c.get("context_score", 0.0) - c.get("phase_penalty", 0.0))
        
    # Sort candidates by combined score descending
    candidates.sort(key=lambda x: x["score_combined"], reverse=True)
    
    # Calculate peak margin
    for i in range(len(candidates)):
        next_score = candidates[i+1]["corr_score"] if i+1 < len(candidates) else 0.0
        candidates[i]["peak_margin"] = candidates[i]["corr_score"] - next_score
        
    return candidates
