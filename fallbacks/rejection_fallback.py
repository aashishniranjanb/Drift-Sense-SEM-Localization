import numpy as np

def evaluate_rejection_fallback(best_candidate: dict, is_ambiguous: bool) -> tuple:
    """
    Aashish Rejection and Confidence Fallback:
    Evaluates whether the candidate match is valid (found=1) or absent/rejected (found=0).
    Calculates a calibrated composite score.
    """
    if best_candidate is None:
        return 0, 0.0
        
    corr = best_candidate.get("corr_score", 0.0)
    psr = best_candidate.get("psr", 0.0)
    margin = best_candidate.get("peak_margin", 0.0)
    context = best_candidate.get("context_score", 0.0)
    phase_penalty = best_candidate.get("phase_penalty", 0.0)
    
    # Calibrated composite score
    composite_score = float(0.40 * corr + 0.30 * context + 0.20 * (psr / 10.0) + 0.10 * margin - phase_penalty)
    composite_score = max(0.0, min(1.0, composite_score))
    
    # Fallback rejection logic:
    found = 1
    if corr < 0.35:
        found = 0
    elif psr < 3.0 and margin < 0.01 and corr < 0.60:
        found = 0
    elif corr < 0.50 and context < 0.20:
        found = 0
        
    return found, composite_score
