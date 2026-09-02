"""
V24-D: Global Lattice Consistency & Adaptive Center Prior
We calculate the 2D pitch of the candidate distribution.
"""
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

def find_lattice_pitches(pool_df):
    """
    Given a pool of candidates, find the dominant periodic X and Y pitches.
    Returns pitch_x, pitch_y.
    """
    if len(pool_df) < 5:
        return 0, 0
        
    # Get all pairwise distances
    pts = pool_df[["cx", "cy"]].values
    dx = np.abs(pts[:, 0:1] - pts[:, 0:1].T)
    dy = np.abs(pts[:, 1:2] - pts[:, 1:2].T)
    
    # Flatten and filter out self-distances and very small noise
    dx = dx[dx > 15]
    dy = dy[dy > 15]
    
    # Histogram voting for pitch
    def dominant_pitch(dists):
        if len(dists) == 0: return 0
        counts, bins = np.histogram(dists, bins=np.arange(0, 800, 5))
        if len(counts) == 0 or np.max(counts) < 5: return 0
        best_bin = np.argmax(counts)
        return bins[best_bin] + 2.5
        
    px = dominant_pitch(dx)
    py = dominant_pitch(dy)
    return px, py

def rank_v24_lattice(pool):
    scores = []
    
    px, py = find_lattice_pitches(pool)
    periodicity_strength = 1.0 if (px > 0 or py > 0) else 0.0
    
    # Adaptive Center Prior
    max_corr = pool["corr_score"].max()
    num_strong_peaks = len(pool[pool["corr_score"] > 0.9 * max_corr])
    
    if periodicity_strength > 0 and num_strong_peaks > 3:
        # Strong periodicity detected -> turn ON center prior heavily
        w_center = 0.15
    else:
        w_center = 0.02
        
    for _, c in pool.iterrows():
        ctx = c["context_128"] if not pd.isna(c["context_128"]) else 0.0
        phase_pen = c["phase_penalty"] if not pd.isna(c["phase_penalty"]) else 0.0
        
        # Base evidence
        base_score = c["corr_score"] + 0.15 * ctx - 0.20 * phase_pen
        
        # Penalty
        d_center = c["dist_to_center"]
        center_penalty = w_center * ((d_center / 250.0) ** 2)
        
        # Lattice residual (if on lattice, penalize less)
        # Assuming the true instance is perfectly on the lattice defined by the max peak
        # For simplicity in this offline test, we'll just rely heavily on the adaptive center.
        
        scores.append(base_score - center_penalty)
        
    return scores

if __name__ == "__main__":
    from test_v24_invariant import eval_ranker
    
    df = pd.read_csv("phase2/V22_CHAMPIONSHIP/results/candidate_pool_features.csv")
    print(f"Loaded {len(df)} candidates.")
    
    for K in [50, 100, 200]:
        c, g, r = eval_ranker(df, rank_v24_lattice, K)
        print(f"V24-D Adaptive Center & Lattice K={K}: {c*100:.2f}% ({r}/{g})")

