import numpy as np
import cv2

def extract_patch_safe(img, x, y, size):
    half = size // 2
    h, w = img.shape[:2]
    
    x = int(round(x))
    y = int(round(y))
    
    y1, y2 = max(0, y - half), min(h, y + half)
    x1, x2 = max(0, x - half), min(w, x + half)
    patch = np.zeros((size, size), dtype=img.dtype)
    py1 = half - (y - y1)
    py2 = half + (y2 - y)
    px1 = half - (x - x1)
    px2 = half + (x2 - x)
    if (py2 > py1) and (px2 > px1) and (y2 > y1) and (x2 > x1):
        patch[py1:py2, px1:px2] = img[y1:y2, x1:x2]
    return patch

def normalize_intensity(patch):
    patch = patch.astype(np.float32)
    if patch.std() == 0: return patch
    return (patch - patch.mean()) / patch.std()

def extract_directional_overlaps(ref_img, search_img, cx, cy, d, patch_size=64):
    ref = extract_patch_safe(ref_img, cx, cy, patch_size)
    search_patches = {
        "C": extract_patch_safe(search_img, cx, cy, patch_size),
        "N": extract_patch_safe(search_img, cx, cy - d, patch_size),
        "S": extract_patch_safe(search_img, cx, cy + d, patch_size),
        "E": extract_patch_safe(search_img, cx + d, cy, patch_size),
        "W": extract_patch_safe(search_img, cx - d, cy, patch_size)
    }
    return ref, search_patches
