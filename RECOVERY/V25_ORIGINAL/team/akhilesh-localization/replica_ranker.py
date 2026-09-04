import numpy as np

def rank_candidates_akhilesh(candidates: list, ref_img=None, search_img=None, est_scale=None, est_theta=None, variant: str = "V18_C") -> list:
    """
    Candidate Replica Discriminator Engine (Phase V18)
    
    Supports:
    - V18_A: Center Prior alone
    - V18_B: Context + Center Prior
    - V18_C: Phase + Context + Center Prior
    - V18_D: Full Handcrafted Multi-Evidence Composite (Winning Heuristic)
    - V18_E: Calibrated Linear Weights
    """
    if len(candidates) == 0:
        return []
        
    ranked = list(candidates)
    
    for c in ranked:
        corr = c.get("corr_score", 0.0)
        psr = c.get("psr", 1.0)
        ctx = c.get("context_combined", c.get("context_128", 0.0))
        phase_pen = c.get("phase_penalty", 0.0)
        phase_res = c.get("phase_residual", 0.0)
        d_center = c.get("dist_to_center", 0.0)
        # Pool-invariant scaling (V24-A+)
        # Assume candidates list is the total pool size K.
        # If pool size is larger, family_population scales. 
        # For K=50, a family of 4 is 8%.
        family_ratio = c.get("family_population", 1) / float(max(1, len(candidates)))
        
        # Adaptive center weight: active only when periodic clustering is detected
        w_center = 0.12 if family_ratio > 0.08 else 0.04
        center_penalty = (d_center / 250.0) ** 2
        
        if variant == "V18_A":
            score = corr - w_center * center_penalty
        elif variant == "V18_B":
            score = corr + 0.15 * ctx - w_center * center_penalty
        elif variant == "V18_C":
            score = corr + 0.15 * ctx - 0.20 * phase_pen - w_center * center_penalty
        elif variant == "V18_D":
            # Normalized multi-evidence composite
            psr_norm = np.clip(psr / 10.0, 0.0, 1.0)
            score = (
                0.40 * corr +
                0.25 * ctx +
                0.15 * psr_norm -
                0.15 * phase_pen -
                0.10 * phase_res -
                w_center * center_penalty
            )
        elif variant == "V18_E":
            # Calibrated Linear Classifier
            score = (
                0.45 * corr +
                0.30 * ctx +
                0.10 * (psr / 10.0) -
                0.20 * phase_pen -
                0.15 * (w_center * center_penalty)
            )
        else:
            score = corr
            
        c["discriminator_score"] = float(score)
        
    ranked.sort(key=lambda x: x["discriminator_score"], reverse=True)
    return ranked
