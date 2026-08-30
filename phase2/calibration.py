import numpy as np

def calibrate_confidence_score(raw_score: float, method: str = "piecewise") -> float:
    """
    Calibrates uncalibrated raw score based on empirical correctness distribution.
    Guarantees strict monotonicity as required by the Phase 2 evaluation criteria.
    """
    s = float(np.clip(raw_score, 0.0, 1.0))
    
    # Monotonic piecewise linear mapping based on validation results
    xp = [0.0, 0.40, 0.65, 0.78, 1.0]
    fp = [0.0, 0.12, 0.42, 0.85, 1.0]
    
    calibrated = np.interp(s, xp, fp)
    return float(np.clip(calibrated, 0.0, 1.0))
