"""STEP 10 -- Global Alignment Discriminator V1 (conservative override).

STEP 9 answer G: V25's features are all local (~100 px). A periodic replica is
locally identical to the true site. The missing signal is *global* -- does one
transformation explain a LARGE region around the candidate, not just its core.

This module computes, for a candidate (x, y, scale, theta):
  - alignment at TWO footprints: 1x template  and  ~3.2x template (captures
    array edges / mat-strip boundaries / routing that a replica gets wrong);
  - concentric ring NCC + core->outer falloff at the large footprint;
  - landmark-constellation consistency (do several reference landmarks re-locate
    under one transform);
  - deltas vs the strongest competitor in the candidate set.

It does NOT re-rank blindly. `decide()` keeps the V25 rank-1 candidate unless
another top-K candidate is BOTH globally much better AND absolutely strong AND
not a falloff replica -- an override, gated so baseline successes cannot break.
No ML.
"""
import numpy as np
import cv2

BIG = 3.2          # large-footprint multiplier over the template size
_RING = None


def _scharr(img):
    f = img.astype(np.float32)
    return cv2.magnitude(cv2.Scharr(f, cv2.CV_32F, 1, 0), cv2.Scharr(f, cv2.CV_32F, 0, 1))


def _ncc(a, b):
    if a.shape != b.shape or a.size == 0:
        return 0.0
    a = a.astype(np.float32) - float(a.mean())
    b = b.astype(np.float32) - float(b.mean())
    sa, sb = float(a.std()), float(b.std())
    if sa < 1e-6 or sb < 1e-6:
        return 0.0
    return float(np.clip((a * b).mean() / (sa * sb), -1.0, 1.0))


def _aligned_ref(ref_img, scale, theta, out_w, out_h, foot=1.0):
    """Render the reference (optionally a `foot`x-larger field-of-view via
    reflect padding) as it should appear in the search frame at (scale, theta)."""
    rh, rw = ref_img.shape[:2]
    if foot > 1.0:
        pad = int(rw * (foot - 1.0) / 2.0)
        ref_img = cv2.copyMakeBorder(ref_img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        rh, rw = ref_img.shape[:2]
    M = cv2.getRotationMatrix2D((rw / 2.0, rh / 2.0), theta, 1.0)
    rot = cv2.warpAffine(ref_img, M, (rw, rh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return cv2.resize(rot, (out_w, out_h), interpolation=cv2.INTER_AREA)


def _win(search_img, cx, cy, w, h):
    sh, sw = search_img.shape[:2]
    x1 = int(round(cx - w / 2.0)); y1 = int(round(cy - h / 2.0))
    x2, y2 = x1 + w, y1 + h
    px1, py1 = max(0, -x1), max(0, -y1)
    px2, py2 = max(0, x2 - sw), max(0, y2 - sh)
    crop = search_img[max(0, y1):min(sh, y2), max(0, x1):min(sw, x2)]
    if px1 or py1 or px2 or py2:
        crop = cv2.copyMakeBorder(crop, py1, py2, px1, px2, cv2.BORDER_REFLECT)
    if crop.shape[:2] != (h, w):
        crop = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)
    return crop


def _rings(h, w):
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot((yy - h / 2.0) / (h / 2.0), (xx - w / 2.0) / (w / 2.0))
    return r <= 0.34, (r > 0.34) & (r <= 0.68), (r > 0.68) & (r <= 1.02)


def _landmarks(ref_al, k=12):
    g = _scharr(ref_al)
    h, w = g.shape
    step = max(20, min(h, w) // 7)
    pts = []
    for yy in range(step, h - step, step):
        for xx in range(step, w - step, step):
            blk = g[yy - step // 2:yy + step // 2, xx - step // 2:xx + step // 2]
            if blk.size:
                pts.append((float(blk.max()), xx, yy))
    pts.sort(reverse=True)
    return [(x, y) for _, x, y in pts[:k]]


def evidence(ref_img, search_img, cx, cy, scale, theta):
    rh, rw = ref_img.shape[:2]
    tw = max(24, int(round(rw / scale)))
    th = max(24, int(round(rh / scale)))

    # footprint 1x
    a1 = _aligned_ref(ref_img, scale, theta, tw, th, foot=1.0)
    w1 = _win(search_img, cx, cy, tw, th)
    small_ncc = _ncc(w1, a1)

    # large footprint
    bw = min(int(tw * BIG), search_img.shape[1])
    bh = min(int(th * BIG), search_img.shape[0])
    aB = _aligned_ref(ref_img, scale, theta, bw, bh, foot=BIG)
    wB = _win(search_img, cx, cy, bw, bh)
    big_ncc = _ncc(wB, aB)
    big_grad = _ncc(_scharr(wB), _scharr(aB))

    core, mid, outer = _rings(bh, bw)
    r_core = _ncc(wB[core], aB[core])
    r_mid = _ncc(wB[mid], aB[mid])
    r_out = _ncc(wB[outer], aB[outer])
    falloff = r_core - r_out

    lms = _landmarks(aB)
    hs = max(12, bw // 24)
    dev = []
    for (lx, ly) in lms:
        y1, y2 = max(0, ly - hs), min(bh, ly + hs)
        x1, x2 = max(0, lx - hs), min(bw, lx + hs)
        tpl = aB[y1:y2, x1:x2]
        if tpl.shape[0] < 8 or tpl.shape[1] < 8:
            continue
        sy1, sy2 = max(0, ly - hs - 10), min(bh, ly + hs + 10)
        sx1, sx2 = max(0, lx - hs - 10), min(bw, lx + hs + 10)
        reg = wB[sy1:sy2, sx1:sx2]
        if reg.shape[0] < tpl.shape[0] or reg.shape[1] < tpl.shape[1]:
            continue
        res = cv2.matchTemplate(reg.astype(np.float32), tpl.astype(np.float32), cv2.TM_CCOEFF_NORMED)
        _, mv, _, ml = cv2.minMaxLoc(res)
        if mv < 0.35:
            continue
        fx = sx1 + ml[0] + tpl.shape[1] / 2.0
        fy = sy1 + ml[1] + tpl.shape[0] / 2.0
        dev.append((fx - lx, fy - ly))
    if dev:
        d = np.array(dev); med = np.median(d, axis=0)
        rms = float(np.sqrt(((d - med) ** 2).sum(1).mean()))
        inl = int((np.sqrt(((d - med) ** 2).sum(1)) <= 3.0).sum())
    else:
        rms, inl = 25.0, 0

    return {"small_ncc": small_ncc, "big_ncc": big_ncc, "big_grad": big_grad,
            "r_core": r_core, "r_mid": r_mid, "r_out": r_out, "falloff": falloff,
            "lm_inliers": inl, "lm_rms": rms, "lm_n": len(lms)}


def gscore(ev):
    """Transparent global-agreement score (higher = more likely the true site)."""
    return (2.2 * ev["big_ncc"] + 1.4 * ev["big_grad"] + 1.6 * ev["r_out"] + 0.6 * ev["r_mid"]
            - 1.8 * ev["falloff"] - 0.045 * ev["lm_rms"] + 0.16 * ev["lm_inliers"])


# --- conservative override ---
OVR_GSCORE_MARGIN = 0.35    # challenger.gscore - anchor.gscore must exceed this
OVR_MIN_BIG_NCC = 0.40      # and challenger must be absolutely strong globally
OVR_MAX_FALLOFF = 0.28      # and not a core-good / outer-bad replica
OVR_ANCHOR_WEAK = 0.30      # only override when the anchor itself is a weak global match


def decide(anchor_ev, challengers):
    """challengers: list of (cand_dict, ev). Returns the chosen cand_dict or None
    (=> keep V25 rank-1). Override only when it is clearly safe."""
    a_g = gscore(anchor_ev)
    if anchor_ev["big_ncc"] >= 0.55 and anchor_ev["falloff"] <= 0.20:
        return None                              # anchor is a strong global match -> never touch
    best = None
    for cand, ev in challengers:
        g = gscore(ev)
        if (g - a_g >= OVR_GSCORE_MARGIN and ev["big_ncc"] >= OVR_MIN_BIG_NCC
                and ev["falloff"] <= OVR_MAX_FALLOFF and anchor_ev["big_ncc"] <= OVR_ANCHOR_WEAK):
            if best is None or g > best[1]:
                best = (cand, g)
    return best[0] if best else None
