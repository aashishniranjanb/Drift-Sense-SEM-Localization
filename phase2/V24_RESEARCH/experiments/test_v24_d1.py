"""
V24-D1 Integrated Ranker Falsification
Integrates row/col consistency and lattice residual into the ranking score.
"""
import pandas as pd
import numpy as np

def get_pitch(d_array):
    if len(d_array) == 0: return 0.0
    counts, bins = np.histogram(d_array, bins=np.arange(0, 800, 5))
    if len(counts) == 0: return 0.0
    best_idx = np.argmax(counts)
    return bins[best_idx] + 2.5

def get_global_phase(vals, pitch, weights=None):
    if pitch == 0: return 0.0
    angles = 2 * np.pi * vals / pitch
    mean_cos = np.average(np.cos(angles), weights=weights)
    mean_sin = np.average(np.sin(angles), weights=weights)
    phase = np.arctan2(mean_sin, mean_cos) * pitch / (2 * np.pi)
    if phase < 0: phase += pitch
    return phase

def get_residual(val, pitch, phase):
    if pitch == 0: return 0.0
    diff = abs(val - phase) % pitch
    return min(diff, pitch - diff)

def eval_v24_d1(df_pool, K):
    total_pairs = 0
    gt_in_pool = 0
    correct_top1 = 0
    
    df_sorted = df_pool.sort_values(["pair_id", "corr_score"], ascending=[True, False])
    
    for pair_id, group in df_sorted.groupby("pair_id"):
        total_pairs += 1
        pool = group.head(K).copy()
        
        if pool["is_correct"].sum() == 0:
            continue
        gt_in_pool += 1
        
        cx_vals = pool["cx"].values
        cy_vals = pool["cy"].values
        corr_vals = pool["corr_score"].values
        
        dx_mat = np.abs(cx_vals[:, None] - cx_vals[None, :])
        dy_mat = np.abs(cy_vals[:, None] - cy_vals[None, :])
        dx = dx_mat[dx_mat > 15]
        dy = dy_mat[dy_mat > 15]
        
        px = get_pitch(dx)
        py = get_pitch(dy)
        
        w = corr_vals ** 3
        phase_x = get_global_phase(cx_vals, px, w)
        phase_y = get_global_phase(cy_vals, py, w)
        
        # We need a metric for how "periodic" the image is to gate the center prior
        # Instead of $O(N^2)$ dilution, use max candidates in a grid line
        max_row_density = max(np.histogram(cy_vals, bins=20)[0]) / K if len(cy_vals) > 0 else 0
        max_col_density = max(np.histogram(cx_vals, bins=20)[0]) / K if len(cx_vals) > 0 else 0
        is_periodic = (max_row_density > 0.15 or max_col_density > 0.15 or px > 0 or py > 0)
        
        scores = []
        for i, c in pool.iterrows():
            ctx = c["context_128"] if not pd.isna(c["context_128"]) else 0.0
            phase_pen = c["phase_penalty"] if not pd.isna(c["phase_penalty"]) else 0.0
            
            # Lattice features
            rx = get_residual(c["cx"], px, phase_x)
            ry = get_residual(c["cy"], py, phase_y)
            lattice_res = np.hypot(rx, ry)
            
            # Row/Col consistency
            r_c = 0.0
            c_c = 0.0
            tol = 15
            if px > 0:
                mask = (np.abs(cy_vals - c["cy"]) < tol) & (np.abs(cx_vals - c["cx"]) > 15)
                rems = np.abs(cx_vals[mask] - c["cx"]) % px
                valid = (rems < tol) | (rems > px - tol)
                r_c = np.sum(corr_vals[mask][valid])
                
            if py > 0:
                mask = (np.abs(cx_vals - c["cx"]) < tol) & (np.abs(cy_vals - c["cy"]) > 15)
                rems = np.abs(cy_vals[mask] - c["cy"]) % py
                valid = (rems < tol) | (rems > py - tol)
                c_c = np.sum(corr_vals[mask][valid])
                
            tot_c = r_c + c_c
            
            # Base V18-C without center
            base_score = c["corr_score"] + 0.15 * ctx - 0.20 * phase_pen
            
            # Adaptive Center
            w_center = 0.12 if (is_periodic and tot_c > 0.5) else 0.04
            center_penalty = w_center * ((c["dist_to_center"] / 250.0) ** 2)
            
            # Combine
            # Lattice residual is small for good candidates (0 to 20)
            # Consistency is large for good candidates (0 to 5)
            # We want to boost consistent candidates, and penalize off-lattice candidates
            
            # Very simple linear combination for falsification
            # Boost score by +0.02 per consistency point
            # Penalize by -0.001 per pixel of lattice residual
            score = base_score - center_penalty + (0.01 * tot_c) - (0.002 * lattice_res)
            
            scores.append(score)
            
        pool["rank_score"] = scores
        best_idx = pool["rank_score"].idxmax()
        if pool.loc[best_idx, "is_correct"] == 1:
            correct_top1 += 1
            
    return correct_top1 / gt_in_pool if gt_in_pool > 0 else 0, gt_in_pool, correct_top1

if __name__ == "__main__":
    df = pd.read_csv("phase2/V22_CHAMPIONSHIP/results/candidate_pool_features.csv")
    print(f"Loaded {len(df)} candidates.")
    
    for K in [50, 100, 200]:
        c, g, r = eval_v24_d1(df, K)
        print(f"V24-D1 (Integrated Lattice+Consistency) K={K}: {c*100:.2f}% ({r}/{g})")
