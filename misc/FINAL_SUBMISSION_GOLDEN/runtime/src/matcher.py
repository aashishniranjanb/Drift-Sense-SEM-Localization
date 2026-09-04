"""Base pose search: coarse-to-fine scale + rotation FFT-NCC, and peak-local
consistency features. Verbatim logic from the frozen V25/V28-C matcher."""
import numpy as np
import cv2
from utils import rotate_image


def coarse_to_fine_scale_search(ref_img, search_img, scale_min=8.0, scale_max=12.0,
                                coarse_step=0.5, fine_step=0.1):
    ref_f = ref_img.astype(np.float32)
    search_f = search_img.astype(np.float32)
    ref_h, ref_w = ref_f.shape[:2]

    best_cs, best_cscale = -1.0, 10.0
    for s in np.arange(scale_min, scale_max + 1e-5, coarse_step):
        tw, th = int(round(ref_w / s)), int(round(ref_h / s))
        if tw < 10 or th < 10 or tw > search_f.shape[1] or th > search_f.shape[0]:
            continue
        tpl = cv2.resize(ref_f, (tw, th), interpolation=cv2.INTER_AREA)
        _, mv, _, _ = cv2.minMaxLoc(cv2.matchTemplate(search_f, tpl, cv2.TM_CCOEFF_NORMED))
        if mv > best_cs:
            best_cs, best_cscale = float(mv), float(s)

    fine_min = max(scale_min, best_cscale - coarse_step)
    fine_max = min(scale_max, best_cscale + coarse_step)
    best = {"best_scale": best_cscale, "best_score": -1.0, "best_template": None,
            "corr_plane": None, "peak_x": 0, "peak_y": 0}
    for s in np.arange(fine_min, fine_max + 1e-5, fine_step):
        tw, th = int(round(ref_w / s)), int(round(ref_h / s))
        if tw < 10 or th < 10 or tw > search_f.shape[1] or th > search_f.shape[0]:
            continue
        tpl = cv2.resize(ref_f, (tw, th), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(search_f, tpl, cv2.TM_CCOEFF_NORMED)
        _, mv, _, ml = cv2.minMaxLoc(res)
        if mv > best["best_score"]:
            best.update(best_scale=float(s), best_score=float(mv), best_template=tpl,
                        corr_plane=res, peak_x=ml[0], peak_y=ml[1])
    return best


def coarse_to_fine_rotation_search(tpl, search_img, angle_min=-5.0, angle_max=5.0, fine_step=0.25):
    tpl_f = tpl.astype(np.float32)
    search_f = search_img.astype(np.float32)
    coarse = [-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0]

    best_cs, best_ca = -1.0, 0.0
    for th in coarse:
        if th < angle_min or th > angle_max:
            continue
        _, mv, _, _ = cv2.minMaxLoc(cv2.matchTemplate(search_f, rotate_image(tpl_f, th), cv2.TM_CCOEFF_NORMED))
        if mv > best_cs:
            best_cs, best_ca = float(mv), float(th)

    fmin, fmax = max(angle_min, best_ca - 1.0), min(angle_max, best_ca + 1.0)
    best = {"best_theta": best_ca, "best_score": -1.0, "rotated_template": None,
            "corr_plane": None, "peak_x": 0, "peak_y": 0}
    for th in np.arange(fmin, fmax + 1e-5, fine_step):
        rt = rotate_image(tpl_f, th)
        res = cv2.matchTemplate(search_f, rt, cv2.TM_CCOEFF_NORMED)
        _, mv, _, ml = cv2.minMaxLoc(res)
        if mv > best["best_score"]:
            best.update(best_theta=float(th), best_score=float(mv), rotated_template=rt,
                        corr_plane=res, peak_x=ml[0], peak_y=ml[1])
    return best


def perform_pose_fallback_search(ref_img, search_img):
    sc = coarse_to_fine_scale_search(ref_img, search_img)
    ro = coarse_to_fine_rotation_search(sc["best_template"], search_img)
    return {"best_scale": float(sc["best_scale"]), "best_theta": float(ro["best_theta"]),
            "best_score": float(ro["best_score"]), "best_template": ro["rotated_template"],
            "corr_plane": ro["corr_plane"]}


def compute_neighborhood_consistency(search_img, template, px, py, pitch_x, pitch_y):
    th, tw = template.shape
    h, w = search_img.shape
    offs = []
    if pitch_x > 0:
        offs += [(pitch_x, 0), (-pitch_x, 0)]
    if pitch_y > 0:
        offs += [(0, pitch_y), (0, -pitch_y)]
    if not offs:
        return 0.0
    scores = []
    for dx, dy in offs:
        nx, ny = int(px + dx), int(py + dy)
        if nx >= 0 and ny >= 0 and nx + tw <= w and ny + th <= h:
            patch = search_img[ny:ny + th, nx:nx + tw]
            if patch.shape == template.shape:
                res = cv2.matchTemplate(patch.astype(np.float32), template.astype(np.float32),
                                        cv2.TM_CCOEFF_NORMED)
                scores.append(float(res[0][0]))
    return float(np.mean(scores)) if scores else 0.0


def compute_gradient_ncc(search_img, template, px, py):
    th, tw = template.shape
    h, w = search_img.shape
    nx, ny = int(px), int(py)
    if nx < 0 or ny < 0 or nx + tw > w or ny + th > h:
        return 0.0
    patch = search_img[ny:ny + th, nx:nx + tw]
    gt = cv2.magnitude(cv2.Sobel(template, cv2.CV_32F, 1, 0, ksize=3),
                       cv2.Sobel(template, cv2.CV_32F, 0, 1, ksize=3))
    gp = cv2.magnitude(cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3),
                       cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3))
    gtu = cv2.normalize(gt, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    gpu = cv2.normalize(gp, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return float(cv2.matchTemplate(gpu, gtu, cv2.TM_CCOEFF_NORMED)[0][0])
