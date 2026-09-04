import numpy as np

def compute_ambiguity_index(candidates: list, scale: float) -> tuple[float, bool]:
    """
    Computes an explicit Ambiguity Index (A) in [0.0, 1.0] based on five factors:
    1. peak_density: Number of candidates with scores within 0.05 of the top candidate.
    2. score_similarity: Margin between Top-1 and Top-2 candidate scores.
    3. lattice_regularity: Spacing match ratio against DRAM/FinFET pitches.
    4. spacing_consistency: Standard deviation of coordinate differences between adjacent candidates.
    5. channel_disagreement: Gradient vs Intensity spatial peak shift.
    
    Returns:
        ambiguity_score (float): Calculated Ambiguity Index.
        is_ambiguous (bool): True if ambiguity >= threshold (0.45), indicating deep search intervention is needed.
    """
    if len(candidates) < 2:
        return 0.0, False

    # Extract scores
    scores = np.array([c["corr_score"] for c in candidates])
    top_score = scores[0]
    
    # 1. Peak Density (normalized count of candidates close to peak)
    close_candidates = np.sum((top_score - scores) < 0.05)
    peak_density = min(1.0, (close_candidates - 1) / 5.0)
    
    # 2. Score Similarity (1.0 - score margin)
    margin = float(candidates[0]["corr_score"] - candidates[1]["corr_score"])
    score_similarity = max(0.0, 1.0 - (margin / 0.05))
    
    # 3. Lattice Regularity
    pitches = np.array([32.0, 36.0, 48.0, 128.0]) / scale
    x_coords = np.array([c["cx"] for c in candidates])
    y_coords = np.array([c["cy"] for c in candidates])
    
    x_diffs = np.abs(np.subtract.outer(x_coords, x_coords))
    y_diffs = np.abs(np.subtract.outer(y_coords, y_coords))
    
    match_count = 0
    total_pairs = 0
    
    for i in range(min(10, len(candidates))):
        for j in range(i + 1, min(10, len(candidates))):
            total_pairs += 1
            dx = x_diffs[i, j]
            dy = y_diffs[i, j]
            x_match = any(abs(dx - p * round(dx / p)) < 2.0 and round(dx / p) > 0 for p in pitches)
            y_match = any(abs(dy - p * round(dy / p)) < 2.0 and round(dy / p) > 0 for p in pitches)
            if x_match or y_match:
                match_count += 1
                
    lattice_regularity = match_count / total_pairs if total_pairs > 0 else 0.0
    
    # 4. Spacing Consistency
    # Compute distances to nearest neighbors
    nn_dists = []
    for i in range(len(candidates)):
        dists = [np.hypot(candidates[i]["cx"] - candidates[j]["cx"], candidates[i]["cy"] - candidates[j]["cy"]) 
                 for j in range(len(candidates)) if i != j]
        if dists:
            nn_dists.append(min(dists))
    spacing_consistency = 0.0
    if len(nn_dists) > 2:
        std_dist = np.std(nn_dists)
        spacing_consistency = max(0.0, 1.0 - (std_dist / 50.0))
        
    # 5. Channel Disagreement (Spatial peak shift for top candidate)
    channel_disagreement = min(1.0, float(candidates[0].get("fft_gradient_score", 0.0) / (candidates[0]["corr_score"] + 1e-6)))
    channel_disagreement = 1.0 - channel_disagreement  # Higher mismatch = higher ambiguity
    
    # Weighted Ambiguity Index Fusion
    # A = 0.25 * density + 0.25 * similarity + 0.20 * lattice + 0.15 * spacing + 0.15 * disagreement
    ambiguity_score = (
        0.25 * peak_density +
        0.25 * score_similarity +
        0.20 * lattice_regularity +
        0.15 * spacing_consistency +
        0.15 * channel_disagreement
    )
    
    # Threshold for intervention decision
    is_ambiguous = ambiguity_score >= 0.45
    return float(ambiguity_score), is_ambiguous
