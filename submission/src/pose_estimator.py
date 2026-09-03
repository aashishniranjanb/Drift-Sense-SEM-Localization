"""Subpixel localization refinement (V24-D3) and V39 surgical pose refinement
(local scale + rotation + paraboloid, with a strict safety gate). Verbatim frozen logic."""
import numpy as np
import cv2
from utils import paraboloid_subpixel


# ----------------------------- V25 / V24-D3 refine -----------------------------
def refine_pose(ref_img, search_img, coarse_scale, coarse_theta, peak_x, peak_y, corr_plane):
    ref_h, ref_w = ref_img.shape[:2]
    tw = int(round(ref_w / coarse_scale))
    th = int(round(ref_h / coarse_scale))
    center_x = float(peak_x) + tw / 2.0
    center_y = float(peak_y) + th / 2.0
    sh, sw = search_img.shape[:2]

    M = cv2.getRotationMatrix2D((ref_w / 2, ref_h / 2), coarse_theta, 1.0)
    rotated = cv2.warpAffine(ref_img, M, (ref_w, ref_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    template = cv2.resize(rotated, (tw, th), interpolation=cv2.INTER_AREA)

    pad = 5
    y1 = int(round(center_y - th / 2)) - pad
    x1 = int(round(center_x - tw / 2)) - pad
    y2 = int(round(center_y + th / 2)) + pad
    x2 = int(round(center_x + tw / 2)) + pad
    if y1 >= 0 and x1 >= 0 and y2 <= sh and x2 <= sw:
        crop = search_img[y1:y2, x1:x2]
        if crop.shape[0] >= th and crop.shape[1] >= tw:
            res = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
            _, _, _, ml = cv2.minMaxLoc(res)
            sp_x, sp_y = paraboloid_subpixel(res, ml[0], ml[1])
            center_x += sp_x - pad
            center_y += sp_y - pad
    return float(center_x), float(center_y), float(coarse_scale), float(coarse_theta)


# ----------------------------- V39 pose refinement -----------------------------
def _scharr(img):
    f = img.astype(np.float32)
    return cv2.magnitude(cv2.Scharr(f, cv2.CV_32F, 1, 0), cv2.Scharr(f, cv2.CV_32F, 0, 1))


def _local_grad_ncc(sg, tg, ml):
    th, tw = tg.shape[:2]
    px, py = ml
    if py + th > sg.shape[0] or px + tw > sg.shape[1] or py < 0 or px < 0:
        return 0.0
    sp = sg[py:py + th, px:px + tw]
    sn = sp - np.mean(sp)
    tn = tg - np.mean(tg)
    ss, ts = np.std(sn), np.std(tn)
    if ss < 1e-6 or ts < 1e-6:
        return 0.0
    return float(np.clip(np.mean(sn * tn) / (ss * ts + 1e-8), -1.0, 1.0))


def refine_scale_local(ref_img, search_img, center_x, center_y, theta0, scale0, pad=4):
    ref_h, ref_w = ref_img.shape[:2]
    sh, sw = search_img.shape[:2]
    facs = [0.9900, 0.9925, 0.9950, 0.9975, 1.0000, 1.0025, 1.0050, 1.0075, 1.0100]
    M = cv2.getRotationMatrix2D((ref_w / 2.0, ref_h / 2.0), theta0, 1.0)
    ref_rot = cv2.warpAffine(ref_img, M, (ref_w, ref_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    max_tw = max(16, int(round(ref_w / (scale0 * 0.9900))))
    max_th = max(16, int(round(ref_h / (scale0 * 0.9900))))
    cp = pad + 4
    y1 = int(round(center_y - max_th / 2.0)) - cp
    x1 = int(round(center_x - max_tw / 2.0)) - cp
    y2 = int(round(center_y + max_th / 2.0)) + cp
    x2 = int(round(center_x + max_tw / 2.0)) + cp
    if y1 < 0 or x1 < 0 or y2 > sh or x2 > sw:
        return scale0, 1.0, 1.0
    crop = search_img[y1:y2, x1:x2]
    crop_g = _scharr(crop)
    best_scale, best_score, score_s0 = scale0, -1.0, -1.0
    for f in facs:
        s = scale0 * f
        tw = max(16, int(round(ref_w / s)))
        th = max(16, int(round(ref_h / s)))
        tcx = center_x - x1
        tcy = center_y - y1
        cy1 = int(round(tcy - th / 2.0)) - pad
        cx1 = int(round(tcx - tw / 2.0)) - pad
        cy2 = int(round(tcy + th / 2.0)) + pad
        cx2 = int(round(tcx + tw / 2.0)) + pad
        if cy1 < 0 or cx1 < 0 or cy2 > crop.shape[0] or cx2 > crop.shape[1]:
            continue
        roi = crop[cy1:cy2, cx1:cx2]
        roi_g = crop_g[cy1:cy2, cx1:cx2]
        if roi.shape[0] < th or roi.shape[1] < tw:
            continue
        tpl = cv2.resize(ref_rot, (tw, th), interpolation=cv2.INTER_AREA)
        tpl_g = _scharr(tpl)
        res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
        _, mv, _, ml = cv2.minMaxLoc(res)
        comb = 0.70 * float(mv) + 0.30 * _local_grad_ncc(roi_g, tpl_g, ml)
        if abs(f - 1.0) < 1e-5:
            score_s0 = comb
        if comb > best_score:
            best_score, best_scale = comb, s
    return float(best_scale), float(best_score), float(score_s0)


def refine_pose_v39(ref_img, search_img, center_x0, center_y0, theta0, scale0, max_displacement_px=1.0):
    ref_h, ref_w = ref_img.shape[:2]
    sh, sw = search_img.shape[:2]
    best_scale, _, _ = refine_scale_local(ref_img, search_img, center_x0, center_y0, theta0, scale0, pad=4)
    tw = max(16, int(round(ref_w / best_scale)))
    th = max(16, int(round(ref_h / best_scale)))
    pad = 4
    y1 = int(round(center_y0 - th / 2.0)) - pad
    x1 = int(round(center_x0 - tw / 2.0)) - pad
    y2 = int(round(center_y0 + th / 2.0)) + pad
    x2 = int(round(center_x0 + tw / 2.0)) + pad
    if y1 < 0 or x1 < 0 or y2 > sh or x2 > sw:
        return center_x0, center_y0, theta0, scale0, {"fallback": True, "reason": "oob", "displacement": 0.0}
    crop = search_img[y1:y2, x1:x2]
    if crop.shape[0] < th or crop.shape[1] < tw:
        return center_x0, center_y0, theta0, scale0, {"fallback": True, "reason": "small", "displacement": 0.0}
    rc = (ref_w / 2.0, ref_h / 2.0)
    best_th, best_ts, best_res, best_ml = theta0, -1.0, None, None
    for d in (-0.5, -0.25, 0.0, 0.25, 0.5):
        thc = theta0 + d
        M = cv2.getRotationMatrix2D(rc, thc, 1.0)
        rot = cv2.warpAffine(ref_img, M, (ref_w, ref_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        tpl = cv2.resize(rot, (tw, th), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(crop, tpl, cv2.TM_CCOEFF_NORMED)
        _, mv, _, ml = cv2.minMaxLoc(res)
        if mv > best_ts:
            best_ts, best_th, best_res, best_ml = float(mv), thc, res, ml
    for d in (-0.10, -0.05, 0.05, 0.10):
        thf = best_th + d
        M = cv2.getRotationMatrix2D(rc, thf, 1.0)
        rot = cv2.warpAffine(ref_img, M, (ref_w, ref_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        tpl = cv2.resize(rot, (tw, th), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(crop, tpl, cv2.TM_CCOEFF_NORMED)
        _, mv, _, ml = cv2.minMaxLoc(res)
        if mv > best_ts:
            best_ts, best_th, best_res, best_ml = float(mv), thf, res, ml
    sp_x, sp_y = paraboloid_subpixel(best_res, best_ml[0], best_ml[1])
    cand_x = center_x0 + (sp_x - pad)
    cand_y = center_y0 + (sp_y - pad)
    disp = float(np.hypot(cand_x - center_x0, cand_y - center_y0))
    if disp > max_displacement_px or best_ts < 0.60:
        return center_x0, center_y0, float(best_th), float(best_scale), {
            "fallback": True, "reason": "gated", "displacement": 0.0, "score": best_ts}
    if disp < 0.5:
        return center_x0, center_y0, float(best_th), float(best_scale), {
            "fallback": False, "reason": "theta_scale_only", "displacement": disp, "score": best_ts}
    return float(cand_x), float(cand_y), float(best_th), float(best_scale), {
        "fallback": False, "reason": "accepted", "displacement": disp, "score": best_ts}
