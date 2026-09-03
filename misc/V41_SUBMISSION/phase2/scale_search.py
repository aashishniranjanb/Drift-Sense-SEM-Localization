import numpy as np
import cv2

def coarse_to_fine_scale_search(ref_img: np.ndarray, search_img: np.ndarray, 
                                scale_min: float = 8.0, scale_max: float = 12.0, 
                                coarse_step: float = 0.5, fine_step: float = 0.1) -> dict:
    """
    Executes a coarse-to-fine scale search in scale range [scale_min, scale_max].
    
    Parameters:
        ref_img: 2D grayscale reference image (typically 1000x1000 physical die view).
        search_img: 2D grayscale search image (1000x1000).
        scale_min: minimum downscaling factor (default 8.0).
        scale_max: maximum downscaling factor (default 12.0).
        coarse_step: step for coarse scale sweep (default 0.5).
        fine_step: step for fine scale sweep (default 0.1).
        
    Returns:
        Dict containing:
            'best_scale': float, estimated scale factor
            'best_score': float, max correlation score achieved
            'best_template': np.ndarray, resampled reference template at best scale
            'corr_plane': np.ndarray, correlation matrix at best scale
            'peak_x': int, x peak index in correlation plane
            'peak_y': int, y peak index in correlation plane
            'coarse_history': list of (scale, score)
    """
    ref_f = ref_img.astype(np.float32)
    search_f = search_img.astype(np.float32)
    ref_h, ref_w = ref_f.shape[:2]
    
    # Coarse search
    coarse_scales = np.arange(scale_min, scale_max + 1e-5, coarse_step)
    best_coarse_score = -1.0
    best_coarse_scale = 10.0
    coarse_history = []
    
    for s in coarse_scales:
        tw = int(round(ref_w / s))
        th = int(round(ref_h / s))
        if tw < 10 or th < 10 or tw > search_f.shape[1] or th > search_f.shape[0]:
            continue
        
        tpl = cv2.resize(ref_f, (tw, th), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(search_f, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        
        coarse_history.append((float(s), float(max_val)))
        if max_val > best_coarse_score:
            best_coarse_score = float(max_val)
            best_coarse_scale = float(s)
            
    # Fine search around best coarse scale
    fine_min = max(scale_min, best_coarse_scale - coarse_step)
    fine_max = min(scale_max, best_coarse_scale + coarse_step)
    fine_scales = np.arange(fine_min, fine_max + 1e-5, fine_step)
    
    best_score = -1.0
    best_scale = best_coarse_scale
    best_corr_plane = None
    best_peak_x = 0
    best_peak_y = 0
    best_template = None
    
    for s in fine_scales:
        tw = int(round(ref_w / s))
        th = int(round(ref_h / s))
        if tw < 10 or th < 10 or tw > search_f.shape[1] or th > search_f.shape[0]:
            continue
            
        tpl = cv2.resize(ref_f, (tw, th), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(search_f, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        
        if max_val > best_score:
            best_score = float(max_val)
            best_scale = float(s)
            best_corr_plane = res
            best_peak_x, best_peak_y = max_loc
            best_template = tpl
            
    return {
        "best_scale": best_scale,
        "best_score": best_score,
        "best_template": best_template,
        "corr_plane": best_corr_plane,
        "peak_x": best_peak_x,
        "peak_y": best_peak_y,
        "coarse_history": coarse_history
    }
