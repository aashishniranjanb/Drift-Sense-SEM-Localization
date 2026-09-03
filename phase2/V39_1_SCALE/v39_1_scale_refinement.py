import cv2
import numpy as np

def compute_scharr_gradient(img: np.ndarray) -> np.ndarray:
    img_f = img.astype(np.float32)
    gx = cv2.Scharr(img_f, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(img_f, cv2.CV_32F, 0, 1)
    return cv2.magnitude(gx, gy)

def compute_local_gradient_ncc(search_crop_grad: np.ndarray, template_grad: np.ndarray, max_loc: tuple[int, int]) -> float:
    th, tw = template_grad.shape[:2]
    px, py = max_loc
    if py + th > search_crop_grad.shape[0] or px + tw > search_crop_grad.shape[1] or py < 0 or px < 0:
        return 0.0
    s_patch = search_crop_grad[py:py+th, px:px+tw]
    s_norm = s_patch - np.mean(s_patch)
    t_norm = template_grad - np.mean(template_grad)
    s_std = np.std(s_norm)
    t_std = np.std(t_norm)
    if s_std < 1e-6 or t_std < 1e-6:
        return 0.0
    ncc = np.mean(s_norm * t_norm) / (s_std * t_std + 1e-8)
    return float(np.clip(ncc, -1.0, 1.0))

def evaluate_scale(ref_img: np.ndarray, search_img: np.ndarray, 
                  center_x: float, center_y: float, theta0: float, scale0: float, pad: int=4) -> tuple[float, float]:
    ref_h, ref_w = ref_img.shape[:2]
    sh, sw = search_img.shape[:2]
    
    ref_center = (ref_w / 2.0, ref_h / 2.0)
    M = cv2.getRotationMatrix2D(ref_center, theta0, 1.0)
    ref_rot = cv2.warpAffine(ref_img, M, (ref_w, ref_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    min_scale = scale0 * 0.985
    max_tw = max(16, int(round(ref_w / min_scale)))
    max_th = max(16, int(round(ref_h / min_scale)))
    
    crop_pad = pad + 4
    y1 = int(round(center_y - max_th / 2.0)) - crop_pad
    x1 = int(round(center_x - max_tw / 2.0)) - crop_pad
    y2 = int(round(center_y + max_th / 2.0)) + crop_pad
    x2 = int(round(center_x + max_tw / 2.0)) + crop_pad
    
    if y1 < 0 or x1 < 0 or y2 > sh or x2 > sw:
        return scale0, -1.0
        
    search_crop = search_img[y1:y2, x1:x2]
    search_crop_grad = compute_scharr_gradient(search_crop)
    
    def score_scale(s_fac: float) -> float:
        s_curr = scale0 * s_fac
        tw = max(16, int(round(ref_w / s_curr)))
        th = max(16, int(round(ref_h / s_curr)))
        
        target_cx_in_crop = center_x - x1
        target_cy_in_crop = center_y - y1
        
        cy1 = int(round(target_cy_in_crop - th / 2.0)) - pad
        cx1 = int(round(target_cx_in_crop - tw / 2.0)) - pad
        cy2 = int(round(target_cy_in_crop + th / 2.0)) + pad
        cx2 = int(round(target_cx_in_crop + tw / 2.0)) + pad
        
        if cy1 < 0 or cx1 < 0 or cy2 > search_crop.shape[0] or cx2 > search_crop.shape[1]:
            return -1.0
            
        roi = search_crop[cy1:cy2, cx1:cx2]
        roi_grad = search_crop_grad[cy1:cy2, cx1:cx2]
        
        if roi.shape[0] < th or roi.shape[1] < tw:
            return -1.0
            
        template = cv2.resize(ref_rot, (tw, th), interpolation=cv2.INTER_AREA)
        template_grad = compute_scharr_gradient(template)
        
        res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        int_ncc = float(max_val)
        
        grad_ncc = compute_local_gradient_ncc(roi_grad, template_grad, max_loc)
        return float(np.clip(0.70 * int_ncc + 0.30 * grad_ncc, -1.0, 1.0))

    coarse_factors = np.arange(0.985, 1.01501, 0.0025)
    best_coarse = 1.0
    best_score = -1.0
    for fac in coarse_factors:
        sc = score_scale(fac)
        if sc > best_score:
            best_score = sc
            best_coarse = fac
            
    fine_factors = np.arange(best_coarse - 0.003, best_coarse + 0.00301, 0.0005)
    best_fine = best_coarse
    for fac in fine_factors:
        sc = score_scale(fac)
        if sc > best_score:
            best_score = sc
            best_fine = fac
            
    return float(scale0 * best_fine), best_score
