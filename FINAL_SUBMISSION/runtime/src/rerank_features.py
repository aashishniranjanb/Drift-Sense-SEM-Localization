import numpy as np
import cv2

def compute_candidate_morphology(corr_plane, peak_x, peak_y, win_r=7):
    """
    Computes peak prominence, curvature, and sharpness directly from corr_plane.
    """
    H, W = corr_plane.shape
    px, py = int(round(peak_x)), int(round(peak_y))

    if px < win_r or px >= W - win_r or py < win_r or py >= H - win_r:
        return {"prominence": 0.0, "curvature": 0.0, "sharpness": 1.0}

    val = float(corr_plane[py, px])
    patch = corr_plane[py - win_r : py + win_r + 1, px - win_r : px + win_r + 1]

    # 1. Prominence: peak value minus the median of the surrounding boundary
    boundary = np.concatenate([patch[0, :], patch[-1, :], patch[:, 0], patch[:, -1]])
    prominence = float(val - np.median(boundary))

    # 2. Curvature (Laplacian): 4 * center - sum of cardinal neighbors
    curv = float(4.0 * val - (corr_plane[py-1, px] + corr_plane[py+1, px] + corr_plane[py, px-1] + corr_plane[py, px+1]))

    # 3. Sharpness: peak value divided by mean of 4 cardinal neighbors
    card_mean = float(corr_plane[py-1, px] + corr_plane[py+1, px] + corr_plane[py, px-1] + corr_plane[py, px+1]) / 4.0
    sharpness = float(val / (card_mean + 1e-5))

    return {
        "prominence": prominence,
        "curvature": curv,
        "sharpness": sharpness
    }

def compute_competitor_features(candidates, cand_idx, pitch_x, pitch_y):
    """
    Computes candidate-vs-competitor features for candidate i.
    """
    c = candidates[cand_idx]
    cx, cy = c["cx"], c["cy"]
    corr = c["corr_score"]

    dists = []
    lat_residuals = []
    other_corrs = []

    px = max(float(pitch_x), 5.0)
    py = max(float(pitch_y), 5.0)

    for j, other in enumerate(candidates):
        if j == cand_idx:
            continue
        dx = abs(other["cx"] - cx)
        dy = abs(other["cy"] - cy)
        d = float(np.hypot(dx, dy))
        dists.append(d)
        other_corrs.append(other["corr_score"])

        # Lattice residual: how close is the offset to an integer multiple of pitch?
        rx = min(dx % px, px - (dx % px)) / px
        ry = min(dy % py, py - (dy % py)) / py
        lat_residuals.append(float(np.hypot(rx, ry)))

    nearest_dist = min(dists) if dists else 100.0
    mean_lat_res = float(np.mean(lat_residuals[:5])) if lat_residuals else 0.5
    top3_corr_gap = corr - float(np.mean(sorted(other_corrs, reverse=True)[:3])) if len(other_corrs) >= 3 else 0.0

    return {
        "nearest_competitor_dist": nearest_dist,
        "lattice_residual": mean_lat_res,
        "top3_corr_gap": top3_corr_gap
    }
