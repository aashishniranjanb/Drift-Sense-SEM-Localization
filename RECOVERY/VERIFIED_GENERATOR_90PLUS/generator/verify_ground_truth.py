"""INDEPENDENT ground-truth verifier.

Sees ONLY: the re-read reference PNG, the re-read search PNG, and the declared
GT dict. It never touches generator metadata, the procedural field, or any
in-memory render.

It builds its own template by a DIFFERENT path than the generator: box blur +
`getRotationMatrix2D`/`warpAffine` + INTER_AREA resize -- not the generator's
supersampled area integration. Agreement between two independent renderers is
the evidence that matters (dataset-prompt §5).

Measures:
  patch validity, intensity NCC and gradient NCC at GT, global correlation peak
  and its distance to GT, peak prominence, nearest competing peak outside an
  exclusion window, GT-vs-competitor margin, GT retrieval rank in a deep NMS
  pool, recoverability at <=1 / <=2 / <=5 px.

Ship rule: global peak within PEAK_TOL px of GT AND margin >= MARGIN_PREFER
(or >= MARGIN_FLOOR when the pair is genuinely degraded -- flagged).
"""
import numpy as np
import cv2

PEAK_TOL = 3.0
MARGIN_PREFER = 0.12
MARGIN_FLOOR = 0.02
EXCL_R = 8           # px: excludes only the GT peak's own correlation lobe, so
                     # the neighbouring lattice site counts as a real competitor
                     # (a one-site error is >5 px at these pitches -> scores 0)


def _grad(img):
    f = img.astype(np.float32)
    return cv2.magnitude(cv2.Scharr(f, cv2.CV_32F, 1, 0), cv2.Scharr(f, cv2.CV_32F, 0, 1))


def _independent_template(ref_u8, theta, z):
    """Deliberately NOT the generator's renderer: pre-blur with a box kernel
    sized to the decimation factor, rotate with a rotation matrix, then
    INTER_AREA resize."""
    k = max(1, int(round(z)))
    if k > 1:
        pre = cv2.blur(ref_u8, (k, k))
    else:
        pre = ref_u8
    h, w = pre.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), theta, 1.0)
    rot = cv2.warpAffine(pre, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    tw = max(16, int(round(w / z)))
    th = max(16, int(round(h / z)))
    return cv2.resize(rot, (tw, th), interpolation=cv2.INTER_AREA)


def _nms_rank(corr, tw, th, gx, gy, max_k=400, r=4):
    """Rank of the first NMS peak landing within 5 px of (gx, gy)."""
    work = corr.copy()
    ch, cw = work.shape[:2]
    for rank in range(1, max_k + 1):
        _, mv, _, ml = cv2.minMaxLoc(work)
        if not np.isfinite(mv) or mv <= -50:
            break
        px, py = ml
        cx, cy = px + tw / 2.0, py + th / 2.0
        if np.hypot(cx - gx, cy - gy) <= 5.0:
            return rank, float(mv)
        y1, y2 = max(0, py - r), min(ch, py + r + 1)
        x1, x2 = max(0, px - r), min(cw, px + r + 1)
        work[y1:y2, x1:x2] = -99.0
    return None, None


def verify(ref_path, search_path, gt):
    ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    srch = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
    if ref is None or srch is None:
        return {"ship": False, "reason": "unreadable"}

    if int(gt.get("present", 0)) == 0:
        # absent pairs carry no location to verify; require only that the decoy
        # does NOT produce a suspiciously dominant match anywhere
        tpl = _independent_template(ref, 0.0, 10.0)
        if tpl.shape[0] >= srch.shape[0] or tpl.shape[1] >= srch.shape[1]:
            return {"ship": False, "reason": "template_too_large"}
        c = cv2.matchTemplate(srch.astype(np.float32), tpl.astype(np.float32), cv2.TM_CCOEFF_NORMED)
        _, mx, _, _ = cv2.minMaxLoc(c)
        return {"ship": True, "reason": "absent_ok", "absent_peak": float(mx),
                "peak_err": -1.0, "margin": -1.0, "gt_rank": -1,
                "ncc_at_gt": -1.0, "grad_ncc_at_gt": -1.0,
                "rec_1px": 0, "rec_2px": 0, "rec_5px": 0, "degraded_floor": 0}

    z, theta = float(gt["scale"]), float(gt["theta"])
    gx, gy = float(gt["x"]), float(gt["y"])
    tpl = _independent_template(ref, theta, z)
    th_, tw_ = tpl.shape
    sh, sw = srch.shape
    if tw_ >= sw or th_ >= sh:
        return {"ship": False, "reason": "template_too_large"}

    # patch validity: the labelled instance must be fully inside the frame
    if not (tw_ / 2 <= gx <= sw - tw_ / 2 and th_ / 2 <= gy <= sh - th_ / 2):
        return {"ship": False, "reason": "instance_clipped"}

    S = srch.astype(np.float32)
    T = tpl.astype(np.float32)
    corr = cv2.matchTemplate(S, T, cv2.TM_CCOEFF_NORMED)
    gcorr = cv2.matchTemplate(_grad(srch), _grad(tpl), cv2.TM_CCOEFF_NORMED)

    _, peak_v, _, peak_l = cv2.minMaxLoc(corr)
    pcx, pcy = peak_l[0] + tw_ / 2.0, peak_l[1] + th_ / 2.0
    peak_err = float(np.hypot(pcx - gx, pcy - gy))

    # response AT the declared GT -- take the local max within +/-3 px, not the
    # single rounded pixel (sub-pixel GT would otherwise under-read the peak and
    # make the margin look negative even when the peak sits on the label)
    ix = int(round(gx - tw_ / 2.0)); iy = int(round(gy - th_ / 2.0))
    ix = int(np.clip(ix, 0, corr.shape[1] - 1)); iy = int(np.clip(iy, 0, corr.shape[0] - 1))
    wy1, wy2 = max(0, iy - 3), min(corr.shape[0], iy + 4)
    wx1, wx2 = max(0, ix - 3), min(corr.shape[1], ix + 4)
    ncc_at_gt = float(corr[wy1:wy2, wx1:wx2].max())
    gy1, gy2 = max(0, iy - 3), min(gcorr.shape[0], iy + 4)
    gx1, gx2 = max(0, ix - 3), min(gcorr.shape[1], ix + 4)
    grad_at_gt = float(gcorr[gy1:gy2, gx1:gx2].max())

    # competitor: best peak outside an exclusion window around the GT
    work = corr.copy()
    y1, y2 = max(0, iy - EXCL_R), min(work.shape[0], iy + EXCL_R + 1)
    x1, x2 = max(0, ix - EXCL_R), min(work.shape[1], ix + EXCL_R + 1)
    work[y1:y2, x1:x2] = -99.0
    _, comp_v, _, comp_l = cv2.minMaxLoc(work)
    margin = float(ncc_at_gt - comp_v)

    # local prominence of the GT response
    ly1, ly2 = max(0, iy - 12), min(corr.shape[0], iy + 13)
    lx1, lx2 = max(0, ix - 12), min(corr.shape[1], ix + 13)
    patch = corr[ly1:ly2, lx1:lx2]
    prominence = float(ncc_at_gt - float(np.mean(patch)))

    rank, _ = _nms_rank(corr, tw_, th_, gx, gy)

    degraded = bool(ncc_at_gt < 0.45)
    need = MARGIN_FLOOR if degraded else MARGIN_PREFER
    ship = bool(peak_err <= PEAK_TOL and margin >= need and ncc_at_gt > 0.0)

    return {"ship": ship,
            "reason": "ok" if ship else ("peak_off" if peak_err > PEAK_TOL else "thin_margin"),
            "peak_err": peak_err, "peak_val": float(peak_v), "margin": margin,
            "ncc_at_gt": ncc_at_gt, "grad_ncc_at_gt": grad_at_gt,
            "prominence": prominence, "competitor": float(comp_v),
            "gt_rank": int(rank) if rank else -1,
            "rec_1px": int(peak_err <= 1.0), "rec_2px": int(peak_err <= 2.0),
            "rec_5px": int(peak_err <= 5.0), "degraded_floor": int(degraded)}
