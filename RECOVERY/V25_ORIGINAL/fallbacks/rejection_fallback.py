import numpy as np
import cv2

def evaluate_rejection_fallback(best_candidate: dict, corr_plane: np.ndarray, 
                                rotated_template: np.ndarray, search_img: np.ndarray) -> tuple:
    """
    Aashish Rejection and Confidence Fallback (V14 Calibrated Presence Engine):
    Uses calibrated multi-evidence scoring combining correlation, wide context (context_128),
    peak margin, PSR, and phase consistency penalties.
    """
    if best_candidate is None:
        return 0, 0.0
        
    corr = float(best_candidate.get("corr_score", 0.0))
    psr = float(best_candidate.get("psr", 0.0))
    margin = float(best_candidate.get("peak_margin", 0.0))
    ctx128 = float(best_candidate.get("context_128", best_candidate.get("context_score", 0.0)))
    phase_res = float(best_candidate.get("phase_residual", 0.0))
    
    # Calibrated multi-evidence presence score (V14 P1)
    composite = float(0.35 * corr + 0.40 * ctx128 + 0.15 * (psr / 10.0) + 0.10 * margin - 0.20 * phase_res)
    calibrated_score = max(0.0, min(1.0, composite))
    
    # Decision threshold (t=0.58 optimized for Set C absence rejection F1)
    found = 1 if calibrated_score >= 0.58 else 0
    
    return int(found), float(calibrated_score)
