"""SEM acquisition + degradation model, applied AFTER the pose transform.

Only photometric operations are applied post-pose, so the label stays exact.
The one geometric effect kept -- raster drift -- returns its displacement at the
target so the caller can add it to the GT (dataset-prompt R5). Nothing else
moves a pixel.

Severity ladder 0..4 (0 = Set-A nominal, 1..4 = Set-B).
"""
import numpy as np
import cv2

SEVERITY = {
    0: dict(dose=340.0, blur=0.55, read=0.006, charge=0.05, streak_p=0.00, streak_i=0.00,
            speckle=0.005, sp=0.0000, vignette=0.03, gamma=1.00, drift=0.0, astig=0.00, contrast=0.03),
    1: dict(dose=180.0, blur=0.85, read=0.012, charge=0.12, streak_p=0.05, streak_i=0.06,
            speckle=0.012, sp=0.0002, vignette=0.08, gamma=1.05, drift=0.30, astig=0.05, contrast=0.07),
    2: dict(dose=95.0, blur=1.15, read=0.020, charge=0.20, streak_p=0.11, streak_i=0.11,
            speckle=0.022, sp=0.0006, vignette=0.14, gamma=1.11, drift=0.55, astig=0.10, contrast=0.11),
    3: dict(dose=52.0, blur=1.55, read=0.030, charge=0.30, streak_p=0.18, streak_i=0.17,
            speckle=0.034, sp=0.0013, vignette=0.20, gamma=1.18, drift=0.85, astig=0.16, contrast=0.16),
    4: dict(dose=28.0, blur=2.00, read=0.042, charge=0.42, streak_p=0.26, streak_i=0.24,
            speckle=0.048, sp=0.0022, vignette=0.27, gamma=1.26, drift=1.20, astig=0.23, contrast=0.22),
}


def _charging(shape, amp, rng):
    h, w = shape
    small = rng.normal(0, 1, (6, 6)).astype(np.float32)
    f = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    f = (f - f.mean()) / (f.std() + 1e-6)
    return 1.0 + amp * f


def _vignette(shape, amp):
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot((yy - h / 2) / (h / 2), (xx - w / 2) / (w / 2))
    return (1.0 - amp * np.clip(r, 0, 1.4) ** 2).astype(np.float32)


def _streaks(img, prob, inten, rng):
    if prob <= 0:
        return img
    h, w = img.shape
    out = img.copy()
    n = int(h * prob)
    for r in rng.choice(h, size=max(0, n), replace=False):
        ln = int(rng.uniform(0.15, 0.7) * w)
        x0 = int(rng.uniform(0, w - ln))
        out[r, x0:x0 + ln] = np.clip(out[r, x0:x0 + ln] + inten * rng.uniform(0.5, 1.5), 0, 1)
    return out


def _astig(img, amp):
    if amp <= 0:
        return img
    k = max(1, int(round(amp * 6)) * 2 + 1)
    return cv2.GaussianBlur(img, (k, 1), 0)


def apply_sem(img01, sev, rng, is_reference=False):
    """img01: float32 [0,1]. Returns (uint8 image, drift_dx, drift_dy).

    The reference is imaged at low severity regardless (it is a clean reference
    acquisition); only the search image carries the severity ladder.
    """
    p = SEVERITY[0 if is_reference else sev]
    img = img01.astype(np.float32).copy()

    # beam PSF
    if p["blur"] > 0:
        k = int(max(3, round(p["blur"] * 4) * 2 + 1))
        img = cv2.GaussianBlur(img, (k, k), p["blur"])
    img = _astig(img, p["astig"])

    # contrast / gamma / charging / vignette
    if p["contrast"] > 0:
        m = img.mean()
        img = np.clip(m + (img - m) * (1.0 + rng.uniform(-p["contrast"], p["contrast"])), 0, 1)
    img = np.clip(img ** p["gamma"], 0, 1)
    img = np.clip(img * _charging(img.shape, p["charge"], rng), 0, 1)
    img = np.clip(img * _vignette(img.shape, p["vignette"]), 0, 1)

    # shot noise (Poisson) + detector read noise
    lam = np.clip(img, 0, 1) * p["dose"]
    img = rng.poisson(lam).astype(np.float32) / max(p["dose"], 1e-6)
    img = np.clip(img + rng.normal(0, p["read"], img.shape), 0, 1)

    # speckle, impulse, charging streaks
    if p["speckle"] > 0:
        img = np.clip(img * (1.0 + rng.normal(0, p["speckle"], img.shape)), 0, 1)
    if p["sp"] > 0:
        m = rng.random(img.shape)
        img[m < p["sp"] / 2] = 0.0
        img[m > 1 - p["sp"] / 2] = 1.0
    img = _streaks(img, p["streak_p"], p["streak_i"], rng)

    # raster drift -- the ONLY geometric op; report its displacement for the GT
    dx = dy = 0.0
    if p["drift"] > 0 and not is_reference:
        dx = float(rng.normal(0, p["drift"] * 0.5))
        dy = float(rng.normal(0, p["drift"] * 0.5))
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    return (np.clip(img, 0, 1) * 255.0).astype(np.uint8), dx, dy
