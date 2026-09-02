import numpy as np
import cv2

def compute_psr(corr_plane: np.ndarray, peak_x: int, peak_y: int, win_r: int = 15) -> tuple[float, float, float]:
    """
    Computes Peak-to-Sidelobe Ratio (PSR) for a correlation peak.
    """
    h, w = corr_plane.shape
    peak_val = corr_plane[peak_y, peak_x]

    y1, y2 = max(0, peak_y - win_r), min(h, peak_y + win_r + 1)
    x1, x2 = max(0, peak_x - win_r), min(w, peak_x + win_r + 1)

    sidelobe = corr_plane[y1:y2, x1:x2].copy()
    sy1, sy2 = max(0, peak_y - 3) - y1, min(h, peak_y + 4) - y1
    sx1, sx2 = max(0, peak_x - 3) - x1, min(w, peak_x + 4) - x1
    sidelobe[sy1:sy2, sx1:sx2] = -999.0

    valid_side = sidelobe[sidelobe > -900.0]
    if len(valid_side) == 0:
        return 10.0, 0.0, 1.0

    mean_side = float(np.mean(valid_side))
    std_side = float(np.std(valid_side)) + 1e-6
    psr = (peak_val - mean_side) / std_side
    return float(psr), mean_side, std_side

def compute_peak_gap(corr_plane: np.ndarray, peak_x: int, peak_y: int, excl_radius: int = 10) -> float:
    """
    Computes Delta-S (difference between 1st peak score and 2nd peak score outside exclusion radius).
    """
    h, w = corr_plane.shape
    val_1st = corr_plane[peak_y, peak_x]
    
    work = corr_plane.copy()
    y1, y2 = max(0, peak_y - excl_radius), min(h, peak_y + excl_radius + 1)
    x1, x2 = max(0, peak_x - excl_radius), min(w, peak_x + excl_radius + 1)
    work[y1:y2, x1:x2] = -999.0
    
    _, max_val_2nd, _, _ = cv2.minMaxLoc(work)
    if max_val_2nd < -900.0:
        return 1.0
    return float(val_1st - max_val_2nd)

def extract_presence_features(corr_plane: np.ndarray, peak_x: int, peak_y: int,
                              ref_template: np.ndarray, search_img: np.ndarray,
                              context_score: float = 0.0, phase_residual: float = 0.0) -> dict:
    """
    Extracts comprehensive presence features including context and phase details.
    """
    psr, _, _ = compute_psr(corr_plane, peak_x, peak_y)
    delta_s = compute_peak_gap(corr_plane, peak_x, peak_y)
    max_score = float(corr_plane[peak_y, peak_x])
    
    return {
        "max_score": max_score,
        "delta_s": delta_s,
        "psr": psr,
        "context_score": float(context_score),
        "phase_residual": float(phase_residual)
    }

def classify_presence(features: dict, score_thresh: float = 0.40) -> tuple[int, float]:
    """
    Evaluates evidence features to determine if reference is present (found=1) or absent (found=0).
    Uses context similarity and phase residuals to reject same-architecture hard negatives (Set C).
    
    Returns:
        found (int): 1 if present, 0 if absent (rejected).
        raw_presence_score (float): Uncalibrated combined presence confidence in [0, 1].
    """
    s = features["max_score"]
    psr = features["psr"]
    context_score = features["context_score"]
    phase_residual = features["phase_residual"]
    
    # Normalized components
    s_term = np.clip(s, 0.0, 1.0)
    psr_term = np.clip(psr / 12.0, 0.0, 1.0)
    context_term = np.clip(context_score, 0.0, 1.0)
    phase_term = np.clip(phase_residual, 0.0, 1.0)
    
    # Decision score fusion
    raw_presence_score = float(0.30 * s_term + 0.20 * psr_term + 0.30 * context_term + 0.20 * phase_term)
    
    # Reject if score is below threshold or context/phase mismatch is too high
    is_absent = (raw_presence_score < score_thresh) or (context_score < 0.30)
    found = 0 if is_absent else 1
    
    return found, raw_presence_score
