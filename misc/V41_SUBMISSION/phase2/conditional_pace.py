import os
import sys
import numpy as np
import cv2
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pace_model import ProcessAwareContextEncoder
from generate_pace_dataset import extract_directional_overlaps, extract_patch_safe, normalize_intensity

DEFAULT_PACE_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "pace_best.pt")

_PACE_MODEL = None
_PACE_DEVICE = None

def load_pace_model(model_path: str = None) -> tuple:
    global _PACE_MODEL, _PACE_DEVICE
    if _PACE_MODEL is not None:
        return _PACE_MODEL, _PACE_DEVICE

    if model_path is None:
        model_path = DEFAULT_PACE_MODEL_PATH

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(model_path):
        return None, device

    try:
        model = ProcessAwareContextEncoder()
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        _PACE_MODEL = model
        _PACE_DEVICE = device
        return model, device
    except Exception as e:
        print(f"Error loading PACE model: {e}")
        return None, device

def extract_canonical_patches(search_img: np.ndarray, cx: float, cy: float, 
                              scale: float, theta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transforms candidate neighborhood in search image back to the canonical frame
    (scale 10.0, rotation 0.0) that the PACE model expects, then extracts patches.
    """
    # Crop a larger neighborhood in search space (e.g. 200x200 pixels)
    crop_size = 250
    sh, sw = search_img.shape[:2]
    
    y1 = max(0, int(round(cy - crop_size // 2)))
    y2 = min(sh, int(round(cy + crop_size // 2)))
    x1 = max(0, int(round(cx - crop_size // 2)))
    x2 = min(sw, int(round(cx + crop_size // 2)))
    
    crop = search_img[y1:y2, x1:x2].copy()
    
    # Pad if bounds exceeded
    pad_y1 = max(0, -int(round(cy - crop_size // 2)))
    pad_y2 = max(0, int(round(cy + crop_size // 2)) - sh)
    pad_x1 = max(0, -int(round(cx - crop_size // 2)))
    pad_x2 = max(0, int(round(cx + crop_size // 2)) - sw)
    if pad_y1 > 0 or pad_y2 > 0 or pad_x1 > 0 or pad_x2 > 0:
        crop = cv2.copyMakeBorder(crop, pad_y1, pad_y2, pad_x1, pad_x2, cv2.BORDER_REFLECT)
        
    # Rotate back by -theta (since search was rotated by theta CCW, we rotate by -theta CW)
    ch, cw = crop.shape[:2]
    center = (cw / 2.0, ch / 2.0)
    M = cv2.getRotationMatrix2D(center, -theta, 1.0)
    rotated = cv2.warpAffine(crop, M, (cw, ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    # Resize back to canonical scale (10x).
    # Since current scale is s, each pixel in search represents s nm.
    # Canonical scale is 10x, so each pixel should represent 10 nm.
    # Resizing factor: s / 10.0
    resize_factor = scale / 10.0
    new_w = int(round(cw * resize_factor))
    new_h = int(round(ch * resize_factor))
    
    canonical_img = cv2.resize(rotated, (new_w, new_h), interpolation=cv2.INTER_AREA if resize_factor < 1.0 else cv2.INTER_CUBIC)
    
    # Extract patches from the center of canonical image
    ccx = new_w / 2.0
    ccy = new_h / 2.0
    
    p64 = normalize_intensity(extract_patch_safe(canonical_img, ccx, ccy, 64))
    p128 = normalize_intensity(extract_patch_safe(canonical_img, ccx, ccy, 128))
    povl = extract_directional_overlaps(canonical_img, ccx, ccy, 32, offset=40)
    
    return p64, p128, povl

def rerank_with_pace(ref_img: np.ndarray, search_img: np.ndarray, 
                     candidates: list, scale: float, theta: float) -> list:
    """
    Reranks candidates using the PACE Process Aware Context Encoder model.
    """
    model, device = load_pace_model()
    if model is None or len(candidates) == 0:
        return candidates

    try:
        # Pre-extract and normalize reference patches in canonical frame (always scale 10, rotation 0)
        ref_100 = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)
        ref_64 = normalize_intensity(cv2.resize(ref_100, (64, 64), interpolation=cv2.INTER_AREA))
        ref_128 = normalize_intensity(cv2.resize(ref_img, (128, 128), interpolation=cv2.INTER_AREA))
        ref_ovl = extract_directional_overlaps(ref_img, 500, 500, 32, offset=200)

        ref_64_t = torch.from_numpy(ref_64).unsqueeze(0).unsqueeze(0).to(device)
        ref_128_t = torch.from_numpy(ref_128).unsqueeze(0).unsqueeze(0).to(device)
        ref_ovl_t = torch.from_numpy(ref_ovl).unsqueeze(0).to(device)

        cand_64_list, cand_128_list, cand_ovl_list, cand_ncc_list = [], [], [], []
        
        for c in candidates:
            p64, p128, povl = extract_canonical_patches(search_img, c["cx"], c["cy"], scale, theta)
            
            cand_64_list.append(torch.from_numpy(p64).unsqueeze(0))
            cand_128_list.append(torch.from_numpy(p128).unsqueeze(0))
            cand_ovl_list.append(torch.from_numpy(povl))
            cand_ncc_list.append(c["corr_score"])

        cand_64_batch = torch.stack(cand_64_list).to(device)
        cand_128_batch = torch.stack(cand_128_list).to(device)
        cand_ovl_batch = torch.stack(cand_ovl_list).to(device)
        cand_ncc_batch = torch.tensor(cand_ncc_list, dtype=torch.float32).to(device)

        with torch.no_grad():
            z_ref = model.forward_encoder(ref_64_t, ref_128_t, ref_ovl_t)
            z_cands = model.forward_encoder(cand_64_batch, cand_128_batch, cand_ovl_batch)
            scores = model(z_ref, z_cands, cand_ncc_batch).cpu().numpy()[0]

        for i, c in enumerate(candidates):
            pace_score = float(scores[i])
            c["pace_score"] = pace_score
            # Combined score: NCC correlation + 0.08 * PACE score
            c["score_combined"] = c["score_combined"] + 0.08 * pace_score

    except Exception as e:
        print(f"PACE re-ranking failed: {e}")
        
    return candidates
