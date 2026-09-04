import numpy as np
import cv2

def crop_and_transform_reference(ref_img: np.ndarray, target_w: int, target_h: int, 
                                 scale: float, theta: float) -> np.ndarray:
    """
    Crops the reference region corresponding to a target window of size target_w x target_h,
    applies rotation theta (degrees) and downscaling, yielding a W x H template matching
    the search image scale.
    """
    ref_h, ref_w = ref_img.shape[:2]
    ref_cx, ref_cy = ref_w / 2.0, ref_h / 2.0
    
    # Scale up target size to reference space
    crop_w = int(round(target_w * scale * 1.1))
    crop_h = int(round(target_h * scale * 1.1))
    
    # Ensure crop falls within reference boundaries
    x1 = max(0, int(ref_cx - crop_w // 2))
    x2 = min(ref_w, int(ref_cx + crop_w // 2))
    y1 = max(0, int(ref_cy - crop_h // 2))
    y2 = min(ref_h, int(ref_cy + crop_h // 2))
    
    ref_crop = ref_img[y1:y2, x1:x2].copy()
    
    # Pad border if crop is smaller than expected
    pad_y1 = max(0, -int(ref_cy - crop_h // 2))
    pad_y2 = max(0, int(ref_cy + crop_h // 2) - ref_h)
    pad_x1 = max(0, -int(ref_cx - crop_w // 2))
    pad_x2 = max(0, int(ref_cx + crop_w // 2) - ref_w)
    if pad_y1 > 0 or pad_y2 > 0 or pad_x1 > 0 or pad_x2 > 0:
        ref_crop = cv2.copyMakeBorder(ref_crop, pad_y1, pad_y2, pad_x1, pad_x2, cv2.BORDER_REFLECT)
        
    # Rotate in reference space
    ch, cw = ref_crop.shape[:2]
    M = cv2.getRotationMatrix2D((cw / 2.0, ch / 2.0), theta, 1.0)
    rotated = cv2.warpAffine(ref_crop, M, (cw, ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    # Crop to exact scaled target size
    sz_w = int(round(target_w * scale))
    sz_h = int(round(target_h * scale))
    sy1 = int(ch // 2 - sz_h // 2)
    sx1 = int(cw // 2 - sz_w // 2)
    cropped_final = rotated[sy1:sy1+sz_h, sx1:sx1+sz_w]
    
    # Downsample to target size
    resized = cv2.resize(cropped_final, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return resized

def compute_ncc(patch_a: np.ndarray, patch_b: np.ndarray) -> float:
    """
    Computes Normalized Cross-Correlation (NCC) between two patches.
    """
    if patch_a.shape != patch_b.shape:
        return 0.0
    a_norm = patch_a.astype(np.float32) - np.mean(patch_a)
    b_norm = patch_b.astype(np.float32) - np.mean(patch_b)
    std_a = np.std(patch_a)
    std_b = np.std(patch_b)
    if std_a < 1e-5 or std_b < 1e-5:
        return 0.0
    return float(np.mean(a_norm * b_norm) / (std_a * std_b))

def verify_candidate_context(ref_img: np.ndarray, search_img: np.ndarray, 
                             cx: float, cy: float, scale: float, theta: float) -> dict:
    """
    Evaluates context similarity at 3 dynamic scales relative to the template size:
    - s_local:  0.35 * Template Size (local cell structure)
    - s_medium: 0.65 * Template Size (intermediate neighborhood)
    - s_global: 0.95 * Template Size (maximum neighborhood within reference bounds)
    """
    sh, sw = search_img.shape[:2]
    
    # Template size in search image space
    t_size = 1000.0 / scale
    
    # Ensure sizes are integers and at least 8 pixels
    size_local = max(8, int(round(0.35 * t_size)))
    size_medium = max(16, int(round(0.65 * t_size)))
    size_global = max(24, int(round(0.95 * t_size)))
    
    scales = [size_local, size_medium, size_global]
    scores = {}
    
    for size in scales:
        sy1 = int(round(cy - size // 2))
        sx1 = int(round(cx - size // 2))
        
        y1, y2 = max(0, sy1), min(sh, sy1 + size)
        x1, x2 = max(0, sx1), min(sw, sx1 + size)
        
        search_patch = search_img[y1:y2, x1:x2].copy()
        
        # Pad search patch if boundary exceeded
        pad_y1 = max(0, -sy1)
        pad_y2 = max(0, (sy1 + size) - sh)
        pad_x1 = max(0, -sx1)
        pad_x2 = max(0, (sx1 + size) - sw)
        if pad_y1 > 0 or pad_y2 > 0 or pad_x1 > 0 or pad_x2 > 0:
            search_patch = cv2.copyMakeBorder(search_patch, pad_y1, pad_y2, pad_x1, pad_x2, cv2.BORDER_REFLECT)
            
        # Transform corresponding reference patch (guaranteed to be inside reference 1000x1000)
        ref_transformed = crop_and_transform_reference(ref_img, size, size, scale, theta)
        
        scores[size] = compute_ncc(search_patch, ref_transformed)
        
    scores["combined"] = 0.20 * scores[size_local] + 0.40 * scores[size_medium] + 0.40 * scores[size_global]
    return {
        "s32": scores[size_local],
        "s64": scores[size_medium],
        "s128": scores[size_global],
        "combined": scores["combined"]
    }
