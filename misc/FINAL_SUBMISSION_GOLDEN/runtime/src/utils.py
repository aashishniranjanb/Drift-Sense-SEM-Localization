"""Shared primitives: image IO, gradients, NMS, subpixel, NCC."""
import numpy as np
import cv2


def load_gray(path):
    """Load an image as single-channel float-safe uint8 grayscale."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def is_true_rgb(bgr):
    """True only if the 3 channels genuinely differ (real colour, not gray-in-3ch)."""
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        return False
    b, g, r = cv2.split(bgr)
    return not (np.array_equal(b, g) and np.array_equal(g, r))


def to_luma(bgr):
    """Rec. 601 luminance from a BGR image."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def scharr_grad(img):
    """Scharr gradient magnitude (float32)."""
    f = img.astype(np.float32) / 255.0
    gx = cv2.Scharr(f, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(f, cv2.CV_32F, 0, 1)
    return cv2.magnitude(gx, gy)


def grad_u8(img):
    """Scharr gradient magnitude rescaled to uint8-range float (for matchTemplate)."""
    g = scharr_grad(img)
    mx = float(g.max())
    if mx > 1e-6:
        g = g / mx
    return (g * 255.0).astype(np.float32)


def rotate_image(img, angle_deg):
    """Rotate about centre, CCW positive, reflect border."""
    if abs(angle_deg) < 1e-4:
        return img.copy()
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def nms_peaks(corr_plane, max_k=200, r=5, floor=-99.0):
    """Greedy non-max suppression on a correlation plane -> list of {px,py,score}."""
    ch, cw = corr_plane.shape[:2]
    work = corr_plane.copy()
    out = []
    for _ in range(max_k):
        _, mv, _, ml = cv2.minMaxLoc(work)
        if mv <= floor or np.isnan(mv):
            break
        px, py = ml
        out.append({"px": int(px), "py": int(py), "score": float(mv)})
        y1, y2 = max(0, py - r), min(ch, py + r + 1)
        x1, x2 = max(0, px - r), min(cw, px + r + 1)
        work[y1:y2, x1:x2] = floor - 900.0
    return out


def peak_prominence(arr, px, py, r):
    """(prominence, z-score) of arr[py,px] vs its (2r+1) neighbourhood."""
    ch, cw = arr.shape[:2]
    y1, y2 = max(0, py - r), min(ch, py + r + 1)
    x1, x2 = max(0, px - r), min(cw, px + r + 1)
    patch = arr[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0, 0.0
    v = float(arr[py, px])
    prom = v - float(np.mean(patch))
    z = prom / (float(np.std(patch)) + 1e-6)
    return prom, z


def competitor_density(cands, px, py, rad):
    """(count within rad, nearest dist, 2nd-nearest dist) among NMS peaks."""
    d = sorted(np.hypot(c["px"] - px, c["py"] - py) for c in cands if (c["px"] != px or c["py"] != py))
    cnt = sum(1 for x in d if x <= rad)
    return cnt, (d[0] if d else 999.0), (d[1] if len(d) > 1 else 999.0)


def paraboloid_subpixel(corr_plane, int_x, int_y):
    """2D quadratic surface fit around an integer peak -> (sub_x, sub_y), clamped +/-0.5."""
    h, w = corr_plane.shape
    if int_x < 2 or int_x >= w - 2 or int_y < 2 or int_y >= h - 2:
        return float(int_x), float(int_y)
    patch = corr_plane[int_y - 2:int_y + 3, int_x - 2:int_x + 3].astype(np.float64)
    yy, xx = np.mgrid[-2:3, -2:3]
    A = np.column_stack([xx.ravel() ** 2, yy.ravel() ** 2, xx.ravel() * yy.ravel(),
                         xx.ravel(), yy.ravel(), np.ones(25)])
    try:
        coeff, _, _, _ = np.linalg.lstsq(A, patch.ravel(), rcond=None)
        a, b, c, d, e, _ = coeff
        denom = 4 * a * b - c ** 2
        if abs(denom) > 1e-6 and a < 0 and b < 0:
            dx = np.clip((c * e - 2 * b * d) / denom, -0.5, 0.5)
            dy = np.clip((c * d - 2 * a * e) / denom, -0.5, 0.5)
            return float(int_x + dx), float(int_y + dy)
    except Exception:
        pass
    return float(int_x), float(int_y)


def ncc(a, b):
    """Zero-mean normalized cross-correlation between two equal-shape patches."""
    if a.shape != b.shape:
        return 0.0
    an = a.astype(np.float32) - float(np.mean(a))
    bn = b.astype(np.float32) - float(np.mean(b))
    sa, sb = float(np.std(a)), float(np.std(b))
    if sa < 1e-5 or sb < 1e-5:
        return 0.0
    return float(np.mean(an * bn) / (sa * sb))
