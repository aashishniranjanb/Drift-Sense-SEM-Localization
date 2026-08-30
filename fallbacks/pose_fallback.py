import numpy as np
import cv2

def rotate_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(angle_deg) < 1e-4:
        return img.copy()
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def perform_pose_fallback_search(ref_img: np.ndarray, search_img: np.ndarray, 
                                 scale_min: float = 8.0, scale_max: float = 12.0, 
                                 angle_min: float = -5.0, angle_max: float = 5.0) -> dict:
    """
    Aashish Fallback Pose Estimator:
    Runs a hierarchical scale and rotation search to recover (scale, theta).
    Supports multi-hypothesis candidate generation to prevent template peak cancellation.
    """
    ref_f = ref_img.astype(np.float32)
    search_f = search_img.astype(np.float32)
    ref_h, ref_w = ref_f.shape[:2]
    
    # 1. Coarse Scale & Rotation Search
    coarse_scales = np.arange(scale_min, scale_max + 1e-5, 0.25)
    coarse_angles = np.arange(angle_min, angle_max + 1e-5, 1.0)
    
    hypotheses = []
    
    for s in coarse_scales:
        tw = int(round(ref_w / s))
        th = int(round(ref_h / s))
        if tw < 10 or th < 10 or tw > search_f.shape[1] or th > search_f.shape[0]:
            continue
        tpl = cv2.resize(ref_f, (tw, th), interpolation=cv2.INTER_AREA)
        
        for theta in coarse_angles:
            rot_tpl = rotate_image(tpl, theta)
            res = cv2.matchTemplate(search_f, rot_tpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            
            hypotheses.append({
                "scale": float(s),
                "theta": float(theta),
                "score": float(max_val),
                "peak_loc": max_loc
            })
            
    # Sort hypotheses by score descending
    hypotheses.sort(key=lambda x: x["score"], reverse=True)
    top_hyps = hypotheses[:3]  # Keep Top-3 hypotheses
    
    # Refine the best hypothesis to fine scale & rotation
    best_hyp = top_hyps[0]
    
    fine_scales = np.arange(max(scale_min, best_hyp["scale"] - 0.2), min(scale_max, best_hyp["scale"] + 0.21), 0.05)
    fine_angles = np.arange(max(angle_min, best_hyp["theta"] - 0.8), min(angle_max, best_hyp["theta"] + 0.81), 0.2)
    
    best_score = -1.0
    best_scale = best_hyp["scale"]
    best_theta = best_hyp["theta"]
    best_template = None
    best_corr = None
    
    for fs in fine_scales:
        tw = int(round(ref_w / fs))
        th = int(round(ref_h / fs))
        tpl = cv2.resize(ref_f, (tw, th), interpolation=cv2.INTER_AREA)
        
        for ftheta in fine_angles:
            rot_tpl = rotate_image(tpl, ftheta)
            res = cv2.matchTemplate(search_f, rot_tpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            
            if max_val > best_score:
                best_score = float(max_val)
                best_scale = float(fs)
                best_theta = float(ftheta)
                best_template = rot_tpl
                best_corr = res
                
    return {
        "best_scale": best_scale,
        "best_theta": best_theta,
        "best_score": best_score,
        "best_template": best_template,
        "corr_plane": best_corr,
        "hypotheses": top_hyps
    }
