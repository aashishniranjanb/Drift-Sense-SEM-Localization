"""Global periodicity estimate from the correlation plane (DRAM/FinFET pitch).
Verbatim frozen logic."""
import numpy as np
from scipy.ndimage import maximum_filter


def estimate_periodicity_from_corr(corr_plane, min_pitch=10, max_pitch=200):
    threshold = float(np.max(corr_plane)) * 0.7
    local_max = maximum_filter(corr_plane, size=5) == corr_plane
    ys, xs = np.where((corr_plane > threshold) & local_max)
    if len(xs) > 500:
        vals = corr_plane[ys, xs]
        top = np.argsort(vals)[-500:]
        ys, xs = ys[top], xs[top]
    if len(xs) < 2:
        return {"pitch_x": 0, "pitch_y": 0, "strength_x": 0.0, "strength_y": 0.0, "mode": "NON_PERIODIC"}
    dx = np.abs(xs[:, None] - xs[None, :]).flatten()
    dy = np.abs(ys[:, None] - ys[None, :]).flatten()
    mx = (dx >= min_pitch) & (dx <= max_pitch) & (dy <= 5)
    my = (dy >= min_pitch) & (dy <= max_pitch) & (dx <= 5)
    px, sx = 0, 0.0
    if np.any(mx):
        hist, edges = np.histogram(dx[mx], bins=np.arange(min_pitch, max_pitch + 2))
        bb = int(np.argmax(hist))
        px, sx = int(edges[bb]), hist[bb] / len(xs)
    py, sy = 0, 0.0
    if np.any(my):
        hist, edges = np.histogram(dy[my], bins=np.arange(min_pitch, max_pitch + 2))
        bb = int(np.argmax(hist))
        py, sy = int(edges[bb]), hist[bb] / len(ys)
    mode = "NON_PERIODIC"
    if sx > 0.5 or sy > 0.5:
        mode = "WEAK"
    if sx > 1.5 or sy > 1.5:
        mode = "STRONG"
    return {"pitch_x": px, "pitch_y": py, "strength_x": sx, "strength_y": sy, "mode": mode}
