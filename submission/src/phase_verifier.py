"""Phase-correlation consistency check at a candidate peak. Verbatim frozen logic."""
import numpy as np
import cv2


def estimator_a_phase_correlation(ref_patch, search_patch):
    h, w = ref_patch.shape[:2]
    sh, sw = search_patch.shape[:2]
    if (h, w) != (sh, sw):
        ref_patch = cv2.resize(ref_patch, (sw, sh), interpolation=cv2.INTER_AREA)
    rf = ref_patch.astype(np.float32)
    sf = search_patch.astype(np.float32)
    hann = cv2.createHanningWindow((sw, sh), cv2.CV_32F)
    rw = (rf - np.mean(rf)) * hann
    sw_ = (sf - np.mean(sf)) * hann
    Ga = np.fft.fft2(rw)
    Gb = np.fft.fft2(sw_)
    cp = Ga * np.conj(Gb)
    r = np.real(np.fft.ifft2(cp / (np.abs(cp) + 1e-7)))
    rs = np.fft.fftshift(r)
    cy, cx = sh // 2, sw // 2
    wr = 5
    sub = rs[max(0, cy - wr):min(sh, cy + wr + 1), max(0, cx - wr):min(sw, cx + wr + 1)]
    _, mv, _, ml = cv2.minMaxLoc(sub)
    return float(ml[0] - wr), float(ml[1] - wr), float(mv)


def verify_phase_consistency(search_img, rot_template, px, py):
    th, tw = rot_template.shape[:2]
    sh, sw = search_img.shape[:2]
    y1, y2 = max(0, int(py)), min(sh, int(py + th))
    x1, x2 = max(0, int(px)), min(sw, int(px + tw))
    crop = search_img[y1:y2, x1:x2]
    if crop.shape != (th, tw):
        return 0.15
    dx, dy, ps = estimator_a_phase_correlation(rot_template, crop)
    disp = float(np.hypot(dx, dy))
    pen = 0.0
    if disp > 2.0:
        pen += float(0.15 * min(1.0, (disp - 2.0) / 4.0))
    if ps < 0.15:
        pen += 0.05
    return float(np.clip(pen, 0.0, 0.20))
