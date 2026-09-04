import numpy as np
import cv2

def compute_neighborhood_consistency(search_img: np.ndarray, template: np.ndarray, px: int, py: int, pitch_x: int, pitch_y: int):
    th, tw = template.shape
    h, w = search_img.shape
    
    scores = []
    offsets = []
    if pitch_x > 0:
        offsets.extend([(pitch_x, 0), (-pitch_x, 0)])
    if pitch_y > 0:
        offsets.extend([(0, pitch_y), (0, -pitch_y)])
        
    if len(offsets) == 0:
        return 0.0
        
    for dx, dy in offsets:
        nx, ny = int(px + dx), int(py + dy)
        if nx >= 0 and ny >= 0 and nx + tw <= w and ny + th <= h:
            patch = search_img[ny:ny+th, nx:nx+tw]
            if patch.shape == template.shape:
                patch_f = patch.astype(np.float32)
                temp_f = template.astype(np.float32)
                res = cv2.matchTemplate(patch_f, temp_f, cv2.TM_CCOEFF_NORMED)
                scores.append(float(res[0][0]))
            
    if len(scores) == 0:
        return 0.0
    return float(np.mean(scores))

def compute_gradient_ncc(search_img: np.ndarray, template: np.ndarray, px: int, py: int):
    th, tw = template.shape
    h, w = search_img.shape
    nx, ny = int(px), int(py)
    
    if nx < 0 or ny < 0 or nx + tw > w or ny + th > h:
        return 0.0
        
    patch = search_img[ny:ny+th, nx:nx+tw]
    
    # Compute Sobel gradients
    sobelx_t = cv2.Sobel(template, cv2.CV_32F, 1, 0, ksize=3)
    sobely_t = cv2.Sobel(template, cv2.CV_32F, 0, 1, ksize=3)
    grad_t = cv2.magnitude(sobelx_t, sobely_t)
    
    sobelx_p = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    sobely_p = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    grad_p = cv2.magnitude(sobelx_p, sobely_p)
    
    # Normalize to uint8 for matchTemplate
    grad_t_u8 = cv2.normalize(grad_t, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    grad_p_u8 = cv2.normalize(grad_p, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    res = cv2.matchTemplate(grad_p_u8, grad_t_u8, cv2.TM_CCOEFF_NORMED)
    return float(res[0][0])

