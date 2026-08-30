import numpy as np
import cv2
from scipy.ndimage import maximum_filter

def extract_adaptive_candidates(corr_plane: np.ndarray, corr_grad: np.ndarray, 
                                tw: int, th: int, max_k: int = 100) -> list:
    """
    Adaptive Peak Detector (V12.1):
    Extracts high-quality candidates using a multi-source union of:
      1. Local Maxima on Intensity Correlation (w=3, 7x7).
      2. Local Maxima on Intensity Correlation (w=4, 9x9).
      3. Iterative NMS with r=5 (preserves periodic replicas while preventing single-peak pixel crowding).
      4. Local Maxima on Gradient Correlation (w=3).
      5. Global Percentile / Thresholded Sites (> 0.70).
      
    Ensures spatial diversity and deduplicates within 3.0 pixels.
    """
    ch, cw = corr_plane.shape[:2]
    candidates_list = []
    
    # 1. Local Maxima (Intensity w=3 & w=4)
    for sz in [7, 9]:
        intensity_max = (maximum_filter(corr_plane, size=sz) == corr_plane) & (corr_plane > 0.05)
        iy, ix = np.where(intensity_max)
        for x, y in zip(ix, iy):
            candidates_list.append({"px": int(x), "py": int(y), "score": float(corr_plane[y, x]), "source": f"local_max_{sz}"})
            
    # 2. Iterative NMS Peaks (r=5)
    work = corr_plane.copy()
    for _ in range(max_k):
        _, max_val, _, max_loc = cv2.minMaxLoc(work)
        if max_val <= 0.05 or np.isnan(max_val):
            break
        px, py = max_loc
        candidates_list.append({"px": int(px), "py": int(py), "score": float(max_val), "source": "nms_r5"})
        y1, y2 = max(0, py - 5), min(ch, py + 6)
        x1, x2 = max(0, px - 5), min(cw, px + 6)
        work[y1:y2, x1:x2] = -999.0
        
    # 3. Local Maxima (Gradient)
    if corr_grad is not None and corr_grad.shape == corr_plane.shape:
        gradient_max = (maximum_filter(corr_grad, size=7) == corr_grad) & (corr_grad > 0.05)
        gy, gx = np.where(gradient_max)
        for x, y in zip(gx, gy):
            candidates_list.append({"px": int(x), "py": int(y), "score": float(corr_plane[y, x]), "source": "gradient_max"})
            
    # 4. Global High-Correlation Threshold (score > 0.70)
    ty, tx = np.where(corr_plane > 0.70)
    for x, y in zip(tx, ty):
        candidates_list.append({"px": int(x), "py": int(y), "score": float(corr_plane[y, x]), "source": "threshold"})
        
    # Sort all candidates by score descending
    candidates_list.sort(key=lambda x: x["score"], reverse=True)
    
    # Spatial Deduplication (NMS radius = 3.0 pixels)
    unique_candidates = []
    for c in candidates_list:
        px, py = c["px"], c["py"]
        cx = px + tw / 2.0
        cy = py + th / 2.0
        
        is_duplicate = False
        for u in unique_candidates:
            if np.hypot(cx - u["cx"], cy - u["cy"]) < 3.0:
                is_duplicate = True
                break
                
        if not is_duplicate:
            unique_candidates.append({
                "peak_x": px,
                "peak_y": py,
                "cx": cx,
                "cy": cy,
                "corr_score": c["score"],
                "source": c["source"]
            })
            if len(unique_candidates) >= max_k:
                break
                
    return unique_candidates
