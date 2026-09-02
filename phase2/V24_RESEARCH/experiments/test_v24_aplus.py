"""
V24-A+ Pool-Invariant Ranker
Replaces raw candidate counts with structural periodicity confidence.
Tests against K=50, 100, 200 to verify pool-invariance.
"""
import pandas as pd
import numpy as np

def get_pitch_and_conf(d_array):
    if len(d_array) == 0: return 0.0, 0.0
    counts, bins = np.histogram(d_array, bins=np.arange(0, 800, 5))
    if len(counts) == 0: return 0.0, 0.0
    best_idx = np.argmax(counts)
    pitch = bins[best_idx] + 2.5
    # Normalize confidence by number of pairs evaluated
    conf = counts[best_idx] / len(d_array) 
    return pitch, conf

def eval_v24_aplus(df_pool, K):
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
        
        dx_mat = np.abs(cx_vals[:, None] - cx_vals[None, :])
        dy_mat = np.abs(cy_vals[:, None] - cy_vals[None, :])
        dx = dx_mat[dx_mat > 15]
        dy = dy_mat[dy_mat > 15]
        
        _, conf_x = get_pitch_and_conf(dx)
        _, conf_y = get_pitch_and_conf(dy)
        
        # Pool-invariant periodicity confidence (max of x and y confidence)
        # For K=50, max confidence is typically ~0.15 for strong periodicity
        # For K=200, it stays stable because both numerator and denominator grow with density
        periodicity_conf = max(conf_x, conf_y)
        
        # Adaptive center weight
        if periodicity_conf > 0.06:
            # Strong periodicity -> use center prior
            # Scale w_center with confidence, up to a max
            w_center = min(0.15, 0.04 + 0.8 * (periodicity_conf - 0.06))
        else:
            w_center = 0.02
            
        scores = []
        for _, c in pool.iterrows():
            ctx = c["context_128"] if not pd.isna(c["context_128"]) else 0.0
            phase_pen = c["phase_penalty"] if not pd.isna(c["phase_penalty"]) else 0.0
            d_center = c["dist_to_center"]
            
            center_penalty = w_center * ((d_center / 250.0) ** 2)
            score = c["corr_score"] + 0.15 * ctx - 0.20 * phase_pen - center_penalty
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
        c, g, r = eval_v24_aplus(df, K)
        print(f"V24-A+ (Periodicity Confidence) K={K}: {c*100:.2f}% ({r}/{g})")
