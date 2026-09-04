"""Multi-scale context verification: how well the neighbourhood around a
candidate matches the reference neighbourhood. Verbatim frozen logic."""
import numpy as np
import cv2
from utils import ncc as compute_ncc


def crop_and_transform_reference(ref_img, target_w, target_h, scale, theta):
    ref_h, ref_w = ref_img.shape[:2]
    rcx, rcy = ref_w / 2.0, ref_h / 2.0
    crop_w = int(round(target_w * scale * 1.1))
    crop_h = int(round(target_h * scale * 1.1))
    x1 = max(0, int(rcx - crop_w // 2)); x2 = min(ref_w, int(rcx + crop_w // 2))
    y1 = max(0, int(rcy - crop_h // 2)); y2 = min(ref_h, int(rcy + crop_h // 2))
    ref_crop = ref_img[y1:y2, x1:x2].copy()
    py1 = max(0, -int(rcy - crop_h // 2)); py2 = max(0, int(rcy + crop_h // 2) - ref_h)
    px1 = max(0, -int(rcx - crop_w // 2)); px2 = max(0, int(rcx + crop_w // 2) - ref_w)
    if py1 or py2 or px1 or px2:
        ref_crop = cv2.copyMakeBorder(ref_crop, py1, py2, px1, px2, cv2.BORDER_REFLECT)
    ch, cw = ref_crop.shape[:2]
    M = cv2.getRotationMatrix2D((cw / 2.0, ch / 2.0), theta, 1.0)
    rot = cv2.warpAffine(ref_crop, M, (cw, ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    sw_ = int(round(target_w * scale)); sh_ = int(round(target_h * scale))
    sy1 = int(ch // 2 - sh_ // 2); sx1 = int(cw // 2 - sw_ // 2)
    return cv2.resize(rot[sy1:sy1 + sh_, sx1:sx1 + sw_], (target_w, target_h), interpolation=cv2.INTER_AREA)


def verify_candidate_context(ref_img, search_img, cx, cy, scale, theta):
    sh, sw = search_img.shape[:2]
    t_size = 1000.0 / scale
    s_local = max(8, int(round(0.35 * t_size)))
    s_med = max(16, int(round(0.65 * t_size)))
    s_glob = max(24, int(round(0.95 * t_size)))
    scores = {}
    for size in (s_local, s_med, s_glob):
        sy1 = int(round(cy - size // 2)); sx1 = int(round(cx - size // 2))
        y1, y2 = max(0, sy1), min(sh, sy1 + size)
        x1, x2 = max(0, sx1), min(sw, sx1 + size)
        patch = search_img[y1:y2, x1:x2].copy()
        py1 = max(0, -sy1); py2 = max(0, (sy1 + size) - sh)
        px1 = max(0, -sx1); px2 = max(0, (sx1 + size) - sw)
        if py1 or py2 or px1 or px2:
            patch = cv2.copyMakeBorder(patch, py1, py2, px1, px2, cv2.BORDER_REFLECT)
        ref_t = crop_and_transform_reference(ref_img, size, size, scale, theta)
        scores[size] = compute_ncc(patch, ref_t)
    combined = 0.20 * scores[s_local] + 0.40 * scores[s_med] + 0.40 * scores[s_glob]
    return {"s32": scores[s_local], "s64": scores[s_med], "s128": scores[s_glob], "combined": combined}
