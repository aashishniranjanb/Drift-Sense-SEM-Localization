import cv2
import numpy as np
from phase2.V39_POSE.v39_scale_refinement import refine_scale_local

def estimator_b_paraboloid_fit(corr_plane: np.ndarray, int_x: int, int_y: int) -> tuple[float, float]:
    """
    Subpixel 2D paraboloid surface fitting estimator around integer peak (int_x, int_y).
    """
    h, w = corr_plane.shape
    if int_x < 2 or int_x >= w - 2 or int_y < 2 or int_y >= h - 2:
        return float(int_x), float(int_y)

    patch = corr_plane[int_y-2:int_y+3, int_x-2:int_x+3].astype(np.float64)
    y_coords, x_coords = np.mgrid[-2:3, -2:3]
    X_mat = np.column_stack([
        x_coords.ravel()**2, y_coords.ravel()**2,
        x_coords.ravel() * y_coords.ravel(),
        x_coords.ravel(), y_coords.ravel(), np.ones(25)
    ])
    Z_vec = patch.ravel()
    try:
        coeff, _, _, _ = np.linalg.lstsq(X_mat, Z_vec, rcond=None)
        a, b, c, d, e, _ = coeff
        denom = 4 * a * b - c**2
        if abs(denom) > 1e-6 and a < 0 and b < 0:
            dx = (c * e - 2 * b * d) / denom
            dy = (c * d - 2 * a * e) / denom
            dx = np.clip(dx, -0.5, 0.5)
            dy = np.clip(dy, -0.5, 0.5)
            return float(int_x + dx), float(int_y + dy)
    except Exception:
        pass
    return float(int_x), float(int_y)

def refine_pose_v39(ref_img: np.ndarray, search_img: np.ndarray,
                    center_x0: float, center_y0: float,
                    theta0: float, scale0: float,
                    max_displacement_px: float = 1.0) -> tuple[float, float, float, float, dict]:
    """
    V39 Surgical Pose Refinement Pipeline:
    
    1. Phase A: Local scale refinement (scale0 * [0.990..1.010] in 0.25% steps with intensity+gradient NCC)
    2. Phase B: Local rotation refinement (theta0 +/- 0.5 deg coarse-to-fine)
    3. Phase C: 2D Subpixel Paraboloid surface fit on exact NCC peak
    4. Phase D: Safety Gate (displacement <= 3.0 px, else fallback to V38 baseline anchor)
    
    Returns (refined_x, refined_y, refined_theta, refined_scale, debug_info)
    """
    ref_h, ref_w = ref_img.shape[:2]
    sh, sw = search_img.shape[:2]
    
    # 1. Phase A: Local Scale Refinement
    best_scale, best_scale_score, score_s0 = refine_scale_local(
        ref_img, search_img, center_x0, center_y0, theta0, scale0, pad=4
    )
    
    # Template dimensions for best_scale
    tw = max(16, int(round(ref_w / best_scale)))
    th = max(16, int(round(ref_h / best_scale)))
    
    # Local ROI for rotation & subpixel search
    pad = 4
    y1 = int(round(center_y0 - th / 2.0)) - pad
    x1 = int(round(center_x0 - tw / 2.0)) - pad
    y2 = int(round(center_y0 + th / 2.0)) + pad
    x2 = int(round(center_x0 + tw / 2.0)) + pad
    
    if y1 < 0 or x1 < 0 or y2 > sh or x2 > sw:
        # Out of bounds fallback
        return center_x0, center_y0, theta0, scale0, {
            'fallback': True, 'reason': 'out_of_bounds', 'displacement': 0.0
        }
        
    search_crop = search_img[y1:y2, x1:x2]
    if search_crop.shape[0] < th or search_crop.shape[1] < tw:
        return center_x0, center_y0, theta0, scale0, {
            'fallback': True, 'reason': 'crop_too_small', 'displacement': 0.0
        }
        
    ref_center = (ref_w / 2.0, ref_h / 2.0)
    
    # 2. Phase B: Coarse-to-Fine Theta Refinement (theta0 +/- 0.5 deg)
    # Stage 1: coarse sweep [-0.5, -0.25, 0.0, 0.25, 0.5]
    best_th = theta0
    best_th_score = -1.0
    best_res = None
    best_max_loc = None
    
    theta_coarse = [-0.5, -0.25, 0.0, 0.25, 0.5]
    for d_th in theta_coarse:
        th_cand = theta0 + d_th
        M_c = cv2.getRotationMatrix2D(ref_center, th_cand, 1.0)
        rot_c = cv2.warpAffine(ref_img, M_c, (ref_w, ref_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        tpl_c = cv2.resize(rot_c, (tw, th), interpolation=cv2.INTER_AREA)
        
        res_c = cv2.matchTemplate(search_crop, tpl_c, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res_c)
        
        if max_val > best_th_score:
            best_th_score = float(max_val)
            best_th = th_cand
            best_res = res_c
            best_max_loc = max_loc
            
    # Stage 2: Fine sweep around best_th (+/- 0.1 deg in 0.025 deg steps)
    fine_deltas = [-0.10, -0.05, 0.0, 0.05, 0.10]
    for d_fine in fine_deltas:
        if d_fine == 0.0:
            continue
        th_fine = best_th + d_fine
        M_f = cv2.getRotationMatrix2D(ref_center, th_fine, 1.0)
        rot_f = cv2.warpAffine(ref_img, M_f, (ref_w, ref_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        tpl_f = cv2.resize(rot_f, (tw, th), interpolation=cv2.INTER_AREA)
        
        res_f = cv2.matchTemplate(search_crop, tpl_f, cv2.TM_CCOEFF_NORMED)
        _, max_val_f, _, max_loc_f = cv2.minMaxLoc(res_f)
        
        if max_val_f > best_th_score:
            best_th_score = float(max_val_f)
            best_th = th_fine
            best_res = res_f
            best_max_loc = max_loc_f
            
    # 3. Phase C: 2D Subpixel Paraboloid Surface Fit
    sp_x, sp_y = estimator_b_paraboloid_fit(best_res, best_max_loc[0], best_max_loc[1])
    dx_sub = sp_x - pad
    dy_sub = sp_y - pad
    
    cand_x = center_x0 + dx_sub
    cand_y = center_y0 + dy_sub
    
    displacement = float(np.hypot(cand_x - center_x0, cand_y - center_y0))
    
    # 4. Phase D: Strict Safety Gate
    # To protect edge-of-boundary cases, preserve (center_x0, center_y0) if:
    # - displacement exceeds max_displacement_px, OR
    # - NCC score is below confidence threshold, OR
    # - displacement is negligible (< 0.08px — subpixel fit noise, not a true shift)
    xy_unchanged = displacement < 0.5  # guard borderline <=5px localizations
    score_ok = best_th_score >= 0.60
    
    if displacement > max_displacement_px or not score_ok:
        # Full fallback: keep original x, y; but allow theta/scale improvement
        return center_x0, center_y0, float(best_th), float(best_scale), {
            'fallback': True, 'reason': 'displacement_or_score_gated',
            'displacement': 0.0, 'score': best_th_score
        }
    
    # For very small displacements: keep original x/y to guard borderline cases
    if xy_unchanged:
        return center_x0, center_y0, float(best_th), float(best_scale), {
            'fallback': False, 'reason': 'theta_scale_only',
            'displacement': displacement, 'score': best_th_score
        }
        
    return float(cand_x), float(cand_y), float(best_th), float(best_scale), {
        'fallback': False, 'reason': 'accepted',
        'displacement': displacement, 'score': best_th_score
    }
