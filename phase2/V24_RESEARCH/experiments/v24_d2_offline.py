"""
V24-D2 Offline Architecture Estimator & Ranker
Tests pool-invariant geometry estimation (using only Top-30 peaks) 
and hierarchical periodic-mode ranking.
"""
import pandas as pd
import numpy as np

def get_pitch_and_conf(d_array, N):
    if len(d_array) == 0: return 0.0, 0.0
    counts, bins = np.histogram(d_array, bins=np.arange(0, 800, 5))
    if len(counts) == 0: return 0.0, 0.0
    best_idx = np.argmax(counts)
    pitch = bins[best_idx] + 2.5
    # Pool-invariant normalization: N is the fixed number of top peaks used (e.g., 30)
    conf = counts[best_idx] / float(N)
    return pitch, conf

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

def eval_v24_d2(df_pool, K=200, N_arch=30):
    total_pairs = 0
    gt_in_pool = 0
    correct_top1 = 0
    
    df_sorted = df_pool.sort_values(["pair_id", "corr_score"], ascending=[True, False])
    
    for pair_id, group in df_sorted.groupby("pair_id"):
        pool = group.head(K).copy()
        if pool["is_correct"].sum() == 0:
            continue
            
        total_pairs += 1
        gt_in_pool += 1
        
        # 1. Architecture Estimation (Strictly from Top N_arch peaks)
        arch_pool = pool.head(N_arch)
        cx_arch = arch_pool["cx"].values
        cy_arch = arch_pool["cy"].values
        corr_arch = arch_pool["corr_score"].values
        
        dx_mat = np.abs(cx_arch[:, None] - cx_arch[None, :])
        dy_mat = np.abs(cy_arch[:, None] - cy_arch[None, :])
        dx = dx_mat[dx_mat > 15]
        dy = dy_mat[dy_mat > 15]
        
        px, conf_x = get_pitch_and_conf(dx, N_arch)
        py, conf_y = get_pitch_and_conf(dy, N_arch)
        p_conf = max(conf_x, conf_y)
        
        # Global Phase anchored ONLY on Top N_arch
        w = corr_arch ** 3
        ph_x = get_global_phase(cx_arch, px, w)
        ph_y = get_global_phase(cy_arch, py, w)
        
        # 2. Ranking Evaluation for all K candidates
        scores = []
        for i, c in pool.iterrows():
            ctx = c["context_128"] if not pd.isna(c["context_128"]) else 0.0
            phase_pen = c["phase_penalty"] if not pd.isna(c["phase_penalty"]) else 0.0
            
            base_score = c["corr_score"] + 0.15 * ctx - 0.20 * phase_pen
            
            # Hierarchical logic
            # Threshold 0.10 for N=30 means at least 3 identical spacings found
            if p_conf > 0.10: 
                # Periodic Mode
                w_center = 0.12
                
                # Compute normalized lattice residual
                rx = get_residual(c["cx"], px, ph_x)
                ry = get_residual(c["cy"], py, ph_y)
                
                # Normalize by pitch to make it scale-invariant (0 to ~0.5)
                nx = rx / px if px > 0 else 0
                ny = ry / py if py > 0 else 0
                norm_res = np.hypot(nx, ny)
                
                # Lattice consistency (bonus for being ON grid)
                lattice_score = -0.05 * norm_res
            else:
                # Non-Periodic Mode
                w_center = 0.02
                lattice_score = 0.0
                
            center_penalty = w_center * ((c["dist_to_center"] / 250.0) ** 2)
            
            scores.append(base_score - center_penalty + lattice_score)
            
        pool["rank_score"] = scores
        best_idx = pool["rank_score"].idxmax()
        if pool.loc[best_idx, "is_correct"] == 1:
            correct_top1 += 1
            
    cond_top1 = correct_top1 / gt_in_pool if gt_in_pool > 0 else 0
    return cond_top1, gt_in_pool, correct_top1

if __name__ == "__main__":
    df = pd.read_csv("phase2/V22_CHAMPIONSHIP/results/candidate_pool_features.csv")
    print(f"Loaded {len(df)} candidates.")
    
    print("\\n=== V24-D2 Architecture Estimator (Arch N=30) ===")
    for K in [50, 100, 200]:
        c, g, r = eval_v24_d2(df, K=K, N_arch=30)
        print(f"Pool K={K:<3} -> Cond Top-1: {c*100:.2f}% ({r}/{g})")
        
    print("\\n=== V24-D2 Ablations (K=200) ===")
    # Sweep lattice weight
    for w_lat in [0.0, 0.02, 0.05, 0.10, 0.15]:
        def eval_sweep(w):
            correct = 0; gt = 0
            df_sorted = df.sort_values(["pair_id", "corr_score"], ascending=[True, False])
            for pair_id, group in df_sorted.groupby("pair_id"):
                pool = group.head(200).copy()
                if pool["is_correct"].sum() == 0: continue
                gt += 1
                arch = pool.head(30)
                dx_mat = np.abs(arch["cx"].values[:, None] - arch["cx"].values[None, :])
                dy_mat = np.abs(arch["cy"].values[:, None] - arch["cy"].values[None, :])
                dx = dx_mat[dx_mat > 15]; dy = dy_mat[dy_mat > 15]
                px, cx = get_pitch_and_conf(dx, 30); py, cy = get_pitch_and_conf(dy, 30)
                p_conf = max(cx, cy)
                ph_x = get_global_phase(arch["cx"].values, px, arch["corr_score"].values**3)
                ph_y = get_global_phase(arch["cy"].values, py, arch["corr_score"].values**3)
                
                scores = []
                for _, c in pool.iterrows():
                    base = c["corr_score"] + 0.15 * (c["context_128"] if not pd.isna(c["context_128"]) else 0) - 0.20 * (c["phase_penalty"] if not pd.isna(c["phase_penalty"]) else 0)
                    if p_conf > 0.30:
                        w_center = 0.12
                        nx = get_residual(c["cx"], px, ph_x)/px if px>0 else 0
                        ny = get_residual(c["cy"], py, ph_y)/py if py>0 else 0
                        lat = -w * np.hypot(nx, ny)
                    else:
                        w_center = 0.02; lat = 0.0
                    scores.append(base - w_center * ((c["dist_to_center"]/250.0)**2) + lat)
                pool["rank_score"] = scores
                if pool.loc[pool["rank_score"].idxmax(), "is_correct"] == 1: correct += 1
            return correct / gt
        print(f"Lattice weight {w_lat:.2f} -> {eval_sweep(w_lat)*100:.2f}%")
