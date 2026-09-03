import numpy as np
import cv2

def rotate_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotates an image around its center by angle_deg degrees (CCW positive).
    Pads with edge reflection/replication to avoid dark border artifacts.
    """
    if abs(angle_deg) < 1e-4:
        return img.copy()
        
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return rotated

def coarse_to_fine_rotation_search(tpl: np.ndarray, search_img: np.ndarray,
                                   angle_min: float = -5.0, angle_max: float = 5.0,
                                   coarse_angles: list = None, fine_step: float = 0.25) -> dict:
    """
    Executes coarse-to-fine rotation search in angle range [angle_min, angle_max] degrees.
    
    Parameters:
        tpl: 2D grayscale template (at optimal scale).
        search_img: 2D grayscale search image.
        angle_min: min angle in degrees (default -5.0).
        angle_max: max angle in degrees (default 5.0).
        coarse_angles: list of angles for coarse sweep (default [-5, -3, -1, 0, 1, 3, 5]).
        fine_step: step size in degrees for fine sweep (default 0.25).
        
    Returns:
        Dict containing:
            'best_theta': float, estimated rotation in degrees
            'best_score': float, max correlation score achieved
            'rotated_template': np.ndarray, rotated template at best angle
            'corr_plane': np.ndarray, correlation plane
            'peak_x': int, x peak index
            'peak_y': int, y peak index
    """
    if coarse_angles is None:
        coarse_angles = [-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0]
        
    tpl_f = tpl.astype(np.float32)
    search_f = search_img.astype(np.float32)
    
    # Coarse search
    best_coarse_score = -1.0
    best_coarse_angle = 0.0
    
    for theta in coarse_angles:
        if theta < angle_min or theta > angle_max:
            continue
        rot_tpl = rotate_image(tpl_f, theta)
        res = cv2.matchTemplate(search_f, rot_tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        
        if max_val > best_coarse_score:
            best_coarse_score = float(max_val)
            best_coarse_angle = float(theta)
            
    # Fine search around best coarse angle
    fine_min = max(angle_min, best_coarse_angle - 1.0)
    fine_max = min(angle_max, best_coarse_angle + 1.0)
    fine_angles = np.arange(fine_min, fine_max + 1e-5, fine_step)
    
    best_score = -1.0
    best_theta = best_coarse_angle
    best_corr_plane = None
    best_peak_x = 0
    best_peak_y = 0
    best_rotated_template = None
    
    for theta in fine_angles:
        rot_tpl = rotate_image(tpl_f, theta)
        res = cv2.matchTemplate(search_f, rot_tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        
        if max_val > best_score:
            best_score = float(max_val)
            best_theta = float(theta)
            best_corr_plane = res
            best_peak_x, best_peak_y = max_loc
            best_rotated_template = rot_tpl
            
    return {
        "best_theta": best_theta,
        "best_score": best_score,
        "rotated_template": best_rotated_template,
        "corr_plane": best_corr_plane,
        "peak_x": best_peak_x,
        "peak_y": best_peak_y
    }
