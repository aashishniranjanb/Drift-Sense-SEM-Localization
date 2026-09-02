import numpy as np
import cv2

def extract_gradient(image: np.ndarray) -> np.ndarray:
    """
    Computes Scharr gradient magnitude map normalized to [0, 1].
    """
    img_f = image.astype(np.float32) / 255.0 if image.dtype == np.uint8 else image.astype(np.float32)
    gx = cv2.Scharr(img_f, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(img_f, cv2.CV_32F, 0, 1)
    g = cv2.magnitude(gx, gy)
    mx = g.max()
    if mx > 1e-6:
        g /= mx
    return g.astype(np.float32)

def compute_channel_consensus(search_img: np.ndarray, search_grad: np.ndarray,
                              rot_template: np.ndarray, rot_template_grad: np.ndarray,
                              px: int, py: int) -> float:
    """
    Evaluates correlation score agreement between Intensity and Gradient channels.
    Returns a consensus penalty in [0.0, 0.20] based on correlation score discrepancy
    and shift between intensity and gradient correlation peaks locally.
    """
    th, tw = rot_template.shape[:2]
    sh, sw = search_img.shape[:2]
    
    # Define local search neighborhood around candidate peak
    win_r = 10
    y1, y2 = max(0, py - win_r), min(sh - th + 1, py + win_r + 1)
    x1, x2 = max(0, px - win_r), min(sw - tw + 1, px + win_r + 1)
    
    search_crop = search_img[y1:y2+th-1, x1:x2+tw-1]
    search_crop_grad = search_grad[y1:y2+th-1, x1:x2+tw-1]
    
    if search_crop.shape[0] < th or search_crop.shape[1] < tw:
        return 0.0
        
    res_int = cv2.matchTemplate(search_crop.astype(np.float32), rot_template.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    res_grad = cv2.matchTemplate(search_crop_grad, rot_template_grad, cv2.TM_CCOEFF_NORMED)
    
    _, max_val_int, _, max_loc_int = cv2.minMaxLoc(res_int)
    _, max_val_grad, _, max_loc_grad = cv2.minMaxLoc(res_grad)
    
    # Calculate peak displacement between intensity and gradient correlation
    dy = max_loc_int[1] - max_loc_grad[1]
    dx = max_loc_int[0] - max_loc_grad[0]
    displacement = np.hypot(dx, dy)
    
    # Penalty scales with displacement and score discrepancy
    score_discrepancy = abs(max_val_int - max_val_grad)
    penalty = 0.10 * min(1.0, displacement / 3.0) + 0.10 * min(1.0, score_discrepancy / 0.3)
    return float(penalty)
