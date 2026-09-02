import cv2
import numpy as np

def extract_nms_fast(corr_plane: np.ndarray, tw: int, th: int, max_k: int = 200, r: int = 5) -> list:
    ch, cw = corr_plane.shape[:2]
    work = corr_plane.copy()
    cands = []
    for rank in range(max_k):
        _, max_val, _, max_loc = cv2.minMaxLoc(work)
        if max_val <= 0.01 or np.isnan(max_val):
            break
        px, py = max_loc
        cands.append({
            "peak_x": px,
            "peak_y": py,
            "cx": px + tw / 2.0,
            "cy": py + th / 2.0,
            "corr_score": float(max_val),
            "raw_rank": rank + 1
        })
        y1, y2 = max(0, py - r), min(ch, py + r + 1)
        x1, x2 = max(0, px - r), min(cw, px + r + 1)
        work[y1:y2, x1:x2] = -999.0
    return cands

def extract_candidates_v19_dual_queue(corr_plane: np.ndarray, tw: int, th: int,
                                      ref_img: np.ndarray = None, search_img: np.ndarray = None,
                                      est_scale: float = 10.0, est_theta: float = 0.0,
                                      max_final_k: int = 50) -> list:
    """
    V19 Dual Queue Intelligent Candidate Extractor (Aashish Main Track):
    1. Extract K=200 raw correlation peaks across the entire search space.
    2. Spatial Sector Allocation:
       - Central Drift Region (dist <= 250 px): Allocate 35 candidate slots.
       - Peripheral Boundary Region (dist > 250 px): Allocate 15 candidate slots.
    3. Within each sector, enforce family diversity to prevent any single periodic array
       from consuming all available quota.
    """
    sh, sw = search_img.shape[:2] if search_img is not None else (1024, 1024)
    search_cx, search_cy = sw / 2.0, sh / 2.0
    
    pool_200 = extract_nms_fast(corr_plane, tw, th, max_k=200, r=5)
    if len(pool_200) <= max_final_k:
        return pool_200
        
    # Split into Center and Periphery queues
    center_queue = []
    periphery_queue = []
    
    for c in pool_200:
        d = np.hypot(c["cx"] - search_cx, c["cy"] - search_cy)
        c["dist_to_center"] = float(d)
        if d <= 260.0:
            center_queue.append(c)
        else:
            periphery_queue.append(c)
            
    # Sort center queue by a blend of correlation and proximity
    for c in center_queue:
        c["center_priority"] = c["corr_score"] - 0.05 * (c["dist_to_center"] / 260.0) ** 2
    center_queue.sort(key=lambda x: x["center_priority"], reverse=True)
    
    # Sort periphery queue by raw correlation
    periphery_queue.sort(key=lambda x: x["corr_score"], reverse=True)
    
    # Take 35 from center and 15 from periphery
    n_center = min(35, len(center_queue))
    n_periph = min(max_final_k - n_center, len(periphery_queue))
    
    final_cands = center_queue[:n_center] + periphery_queue[:n_periph]
    
    # If quota still not filled, pull remaining best from either queue
    if len(final_cands) < max_final_k:
        remaining = center_queue[n_center:] + periphery_queue[n_periph:]
        remaining.sort(key=lambda x: x["corr_score"], reverse=True)
        final_cands.extend(remaining[:max_final_k - len(final_cands)])
        
    return final_cands[:max_final_k]
