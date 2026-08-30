import numpy as np
import cv2

def estimator_a_phase_correlation(ref_patch: np.ndarray, search_patch: np.ndarray) -> tuple[float, float, float]:
    """
    Subpixel phase correlation estimator between reference patch and search patch.
    """
    h, w = ref_patch.shape[:2]
    sh, sw = search_patch.shape[:2]
    if (h, w) != (sh, sw):
        ref_patch = cv2.resize(ref_patch, (sw, sh), interpolation=cv2.INTER_AREA)

    ref_f = ref_patch.astype(np.float32)
    search_f = search_patch.astype(np.float32)
    hann = cv2.createHanningWindow((sw, sh), cv2.CV_32F)
    ref_win = (ref_f - np.mean(ref_f)) * hann
    search_win = (search_f - np.mean(search_f)) * hann

    G_a = np.fft.fft2(ref_win)
    G_b = np.fft.fft2(search_win)
    cross_power = G_a * np.conj(G_b)
    magnitude = np.abs(cross_power) + 1e-7
    r = np.real(np.fft.ifft2(cross_power / magnitude))
    r_shift = np.fft.fftshift(r)

    cy, cx = sh // 2, sw // 2
    win_r = 5
    sub_r = r_shift[max(0, cy - win_r):min(sh, cy + win_r + 1),
                    max(0, cx - win_r):min(sw, cx + win_r + 1)]
    _, max_val, _, max_loc = cv2.minMaxLoc(sub_r)
    return float(max_loc[0] - win_r), float(max_loc[1] - win_r), float(max_val)


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


def refine_pose(ref_img: np.ndarray, search_img: np.ndarray, 
                coarse_scale: float, coarse_theta: float, 
                peak_x: int, peak_y: int, corr_plane: np.ndarray) -> tuple[float, float, float, float]:
    """
    Refines subpixel (x, y) center location, scale, and rotation.
    
    Returns:
        (x_center, y_center, refined_scale, refined_theta)
    """
    ref_h, ref_w = ref_img.shape[:2]
    tw = int(round(ref_w / coarse_scale))
    th = int(round(ref_h / coarse_scale))
    
    # 1. Paraboloid fit for (x, y)
    sub_px, sub_py = estimator_b_paraboloid_fit(corr_plane, peak_x, peak_y)
    
    # Target center in search image coordinates (1-indexed or 0-indexed center)
    center_x = sub_px + tw / 2.0
    center_y = sub_py + th / 2.0
    
    # 2. Phase correlation consensus check
    sh, sw = search_img.shape[:2]
    cy_i, cx_i = int(round(center_y)), int(round(center_x))
    y1, y2 = max(0, cy_i - th // 2), min(sh, cy_i + th // 2)
    x1, x2 = max(0, cx_i - tw // 2), min(sw, cx_i + tw // 2)
    
    search_crop = search_img[y1:y2, x1:x2]
    if search_crop.shape[0] > 10 and search_crop.shape[1] > 10:
        tpl_res = cv2.resize(ref_img, (search_crop.shape[1], search_crop.shape[0]), interpolation=cv2.INTER_AREA)
        dx_p, dy_p, p_score = estimator_a_phase_correlation(tpl_res, search_crop)
        if abs(dx_p) < 2.0 and abs(dy_p) < 2.0:
            center_x += 0.5 * dx_p
            center_y += 0.5 * dy_p
            
    return float(center_x), float(center_y), float(coarse_scale), float(coarse_theta)
