import numpy as np

def calculate_confidence(ev):
    """
    ev contains:
    - top1_score: Overall V25 structural score
    - margin: Gap between top1 and top2 scores
    - top1_corr: Base template correlation
    - top1_ctx: Context match score
    - top1_neigh: Neighborhood consistency
    - top1_grad: Gradient phase agreement
    - mode_strong: Number of peaks in the cluster
    
    Returns P_present, P_correct_given_present, P_correct
    """
    # 1. P_present: Does this look like a real match at all?
    # Based heavily on structural scores.
    base_p = ev.get('top1_score', 0.0)
    
    # 2. P_correct_given_present: Is this the TRUE match, or a periodic replica?
    # Margin and neighborhood are key for disambiguating replicas.
    margin = ev.get('margin', 0.0)
    neigh = ev.get('top1_neigh', 0.0)
    ctx = ev.get('top1_ctx', 0.0)
    
    # Normalize margin (margin > 0.05 is usually very confident)
    margin_norm = np.clip(margin / 0.05, 0.0, 1.0)
    
    # Context and neighborhood consistency
    structural_consensus = (0.5 * neigh + 0.5 * ctx)
    
    p_correct_given = 0.4 * margin_norm + 0.6 * structural_consensus
    
    p_present = np.clip(base_p, 0.0, 1.0)
    p_correct = p_present * p_correct_given
    
    return float(p_present), float(p_correct_given), float(p_correct)
