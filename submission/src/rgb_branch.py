"""Optical / RGB bonus path: Rec. 601 luminance -> dual-channel (intensity U
gradient) FFT-NCC candidate union -> V25 subpixel refinement. Verbatim frozen
logic from the RGB bonus submission."""
import numpy as np
import cv2

from matcher import perform_pose_fallback_search
from pose_estimator import refine_pose

RGB_FOUND_THRESHOLD = 0.4


def _grad_u8(img):
    f = img.astype(np.float32) / 255.0
    gx = cv2.Scharr(f, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(f, cv2.CV_32F, 0, 1)
    g = cv2.magnitude(gx, gy)
    mx = float(g.max())
    if mx > 1e-6:
        g = g / mx
    return (g * 255.0).astype(np.uint8)


def run_rgb_localization(ref_color, search_color):
    ref_y = cv2.cvtColor(ref_color, cv2.COLOR_BGR2GRAY)
    search_y = cv2.cvtColor(search_color, cv2.COLOR_BGR2GRAY)

    pose = perform_pose_fallback_search(ref_y, search_y)
    search_g = _grad_u8(search_y)
    template_g = _grad_u8(pose["best_template"])
    corr_g = cv2.matchTemplate(search_g.astype(np.float32), template_g.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    corr_union = np.maximum(pose["corr_plane"], corr_g)
    _, max_val, _, max_loc = cv2.minMaxLoc(corr_union)

    rx, ry, _, _ = refine_pose(ref_y, search_y, pose["best_scale"], pose["best_theta"],
                               max_loc[0], max_loc[1], corr_union)
    found = 1 if max_val > RGB_FOUND_THRESHOLD else 0
    if found == 0:
        return {"x": 0.0, "y": 0.0, "theta": 0.0, "scale": 0.0, "found": 0, "score": 0.0}
    return {"x": float(rx), "y": float(ry), "theta": float(pose["best_theta"]),
            "scale": float(pose["best_scale"]), "found": 1, "score": float(max_val)}
