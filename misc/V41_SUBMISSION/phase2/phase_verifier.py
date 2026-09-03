import numpy as np
import cv2
from pose_refinement import estimator_a_phase_correlation

def verify_phase_consistency(search_img: np.ndarray, rot_template: np.ndarray, 
                              px: int, py: int) -> float:
    """
    Computes local phase correlation consistency for a candidate peak at px, py.
    Returns a penalty in [0.0, 0.20] if phase correlation displacement is large
    or phase peak score is extremely low.
    """
    th, tw = rot_template.shape[:2]
    sh, sw = search_img.shape[:2]
    
    # Crop candidate search patch
    y1, y2 = max(0, int(py)), min(sh, int(py + th))
    x1, x2 = max(0, int(px)), min(sw, int(px + tw))
    search_crop = search_img[y1:y2, x1:x2]
    
    if search_crop.shape != (th, tw):
        return 0.15
        
    dx_p, dy_p, p_score = estimator_a_phase_correlation(rot_template, search_crop)
    displacement = np.hypot(dx_p, dy_p)
    
    penalty = 0.0
    if displacement > 2.0:
        penalty += float(0.15 * min(1.0, (displacement - 2.0) / 4.0))
    if p_score < 0.15:
        penalty += 0.05
        
    return float(np.clip(penalty, 0.0, 0.20))
