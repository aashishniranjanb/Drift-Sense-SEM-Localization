import numpy as np
import cv2

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

def refine_pose_v38_safe(ref_img: np.ndarray, search_img: np.ndarray, 
                         coarse_scale: float, coarse_theta: float, 
                         peak_x: int, peak_y: int, corr_plane: np.ndarray) -> tuple[float, float, float, float]:
    """
    Safe Subpixel & Pose Optimizer (V38-Safe):
    Preserves exact V25 spatial anchor (center_x, center_y) while applying:
    1. Exact spatial NCC peak search (+/- 3px)
    2. 2D Paraboloid Subpixel Fit
    3. Fine Rotation Alignment (+/- 0.5 deg fine step) without altering candidate anchor.
    """
    ref_h, ref_w = ref_img.shape[:2]
    sh, sw = search_img.shape[:2]
    
    tw = int(round(ref_w / coarse_scale))
    th = int(round(ref_h / coarse_scale))
    
    center_x = float(peak_x) + tw / 2.0
    center_y = float(peak_y) + th / 2.0
    
    # 1. Exact spatial NCC search around peak_x, peak_y (+/- 5px)
    ref_center = (ref_w / 2.0, ref_h / 2.0)
    M0 = cv2.getRotationMatrix2D(ref_center, coarse_theta, 1.0)
    rotated0 = cv2.warpAffine(ref_img, M0, (ref_w, ref_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    template0 = cv2.resize(rotated0, (tw, th), interpolation=cv2.INTER_AREA)
    
    pad = 5
    y1 = int(round(center_y - th / 2.0)) - pad
    x1 = int(round(center_x - tw / 2.0)) - pad
    y2 = int(round(center_y + th / 2.0)) + pad
    x2 = int(round(center_x + tw / 2.0)) + pad
    
    if y1 >= 0 and x1 >= 0 and y2 <= sh and x2 <= sw:
        search_crop = search_img[y1:y2, x1:x2]
        if search_crop.shape[0] >= th and search_crop.shape[1] >= tw:
            res = cv2.matchTemplate(search_crop, template0, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(res)
            
            sp_x, sp_y = estimator_b_paraboloid_fit(res, max_loc[0], max_loc[1])
            dx_sub = sp_x - pad
            dy_sub = sp_y - pad
            
            center_x += dx_sub
            center_y += dy_sub
            
            # Fine theta search around coarse_theta
            best_th = coarse_theta
            best_th_score = float(res[max_loc[1], max_loc[0]])
            for d_th in [-0.5, -0.25, 0.25, 0.5]:
                th_cand = coarse_theta + d_th
                M_c = cv2.getRotationMatrix2D(ref_center, th_cand, 1.0)
                rot_c = cv2.warpAffine(ref_img, M_c, (ref_w, ref_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                tpl_c = cv2.resize(rot_c, (tw, th), interpolation=cv2.INTER_AREA)
                res_c = cv2.matchTemplate(search_crop, tpl_c, cv2.TM_CCOEFF_NORMED)
                val_c = float(res_c[max_loc[1], max_loc[0]])
                if val_c > best_th_score:
                    best_th_score = val_c
                    best_th = th_cand
            coarse_theta = best_th

    return float(center_x), float(center_y), float(coarse_theta), float(coarse_scale)

refine_pose_v38 = refine_pose_v38_safe

