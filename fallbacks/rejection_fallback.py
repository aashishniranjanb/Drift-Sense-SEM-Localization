import numpy as np
import cv2

# Import robust presence classification from phase2 baseline
import sys
sys.path.append("phase2")
from inference_phase2 import extract_presence_features, classify_presence

def evaluate_rejection_fallback(best_candidate: dict, corr_plane: np.ndarray, 
                                rotated_template: np.ndarray, search_img: np.ndarray) -> tuple:
    """
    Aashish Rejection and Confidence Fallback:
    Evaluates whether the candidate match is valid (found=1) or absent/rejected (found=0)
    using the full robust baseline presence classification.
    """
    if best_candidate is None:
        return 0, 0.0
        
    px = best_candidate.get("peak_x", 0)
    py = best_candidate.get("peak_y", 0)
    context_score = best_candidate.get("context_score", 0.0)
    phase_residual = best_candidate.get("phase_residual", 0.0)
    
    # Extract presence features
    presence_feats = extract_presence_features(
        corr_plane, px, py, rotated_template, search_img,
        context_score=context_score, phase_residual=phase_residual
    )
    
    # Classify presence
    found, raw_presence_score = classify_presence(presence_feats)
    
    # Calibrated confidence score
    if found == 0:
        decision_confidence = 1.0 - raw_presence_score
    else:
        decision_confidence = raw_presence_score
        
    return int(found), float(decision_confidence)
