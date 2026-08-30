import numpy as np
import cv2
from scipy.ndimage import maximum_filter

def extract_adaptive_candidates(corr_plane: np.ndarray, corr_grad: np.ndarray, 
                                tw: int, th: int, max_k: int = 50) -> list:
    """
    Adaptive Peak Detector (V11.1):
    Extracts high-quality candidates using a union of:
      1. Local Maxima on Intensity Correlation (w=4).
      2. Local Maxima on Gradient Correlation (w=4).
      3. Global Thresholded High-correlation sites.
      
    Ensures spatial diversity and removes duplicates within 3.0 pixels.
    """
    ch, cw = corr_plane.shape[:2]
    
    # 1. Local Maxima (Intensity)
    size_int = 9  # 9x9 neighborhood (w=4)
    intensity_max = (maximum_filter(corr_plane, size=size_int) == corr_plane) & (corr_plane > 0.05)
    iy, ix = np.where(intensity_max)
    candidates_list = [{"px": int(x), "py": int(y), "score": float(corr_plane[y, x]), "source": "intensity"} for x, y in zip(ix, iy)]
    
    # 2. Local Maxima (Gradient)
    size_grad = 9
    gradient_max = (maximum_filter(corr_grad, size=size_grad) == corr_grad) & (corr_grad > 0.05)
    gy, gx = np.where(gradient_max)
    for x, y in zip(gx, gy):
        candidates_list.append({"px": int(x), "py": int(y), "score": float(corr_plane[y, x]), "source": "gradient"})
        
    # 3. Global High-Correlation Threshold (score > 0.75)
    ty, tx = np.where(corr_plane > 0.75)
    for x, y in zip(tx, ty):
        candidates_list.append({"px": int(x), "py": int(y), "score": float(corr_plane[y, x]), "source": "threshold"})
        
    # Sort candidates by correlation score descending
    candidates_list.sort(key=lambda x: x["score"], reverse=True)
    
    # Spatial NMS / Deduplication (NMS radius = 3 pixels)
    unique_candidates = []
    for c in candidates_list:
        px, py = c["px"], c["py"]
        cx = px + tw / 2.0
        cy = py + th / 2.0
        
        # Check if already covered
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
