"""
V24-A & V24-B: Pool-Invariant Ranker Evaluation
Tests ranking algorithms directly on the cached candidate_pool_features.csv
"""
import pandas as pd
import numpy as np
import os

def eval_ranker(df_pool, ranker_func, K):
    """
    Evaluates a ranker on a subset of Top-K extracted candidates per pair.
    df_pool: DataFrame of all 200 candidates for all pairs, sorted natively by raw corr_score (implied by extraction)
    K: How many top candidates to consider per pair (simulating pool extraction size)
    """
    total_pairs = 0
    gt_in_pool = 0
    correct_top1 = 0
    
    # We assume candidates per pair are ordered by raw extraction score (e.g., peak_margin or corr_score).
    # Actually, let's explicitly sort by corr_score just to simulate the initial NMS output order
    df_sorted = df_pool.sort_values(["pair_id", "corr_score"], ascending=[True, False])
    
    for pair_id, group in df_sorted.groupby("pair_id"):
        total_pairs += 1
        # Truncate to top K
        pool = group.head(K).copy()
        
        # Is GT in this pool?
        if pool["is_correct"].sum() == 0:
            continue
        gt_in_pool += 1
        
        # Score candidates
        scores = ranker_func(pool)
        pool["rank_score"] = scores
        
        # Evaluate Top-1
        best_idx = pool["rank_score"].idxmax()
        if pool.loc[best_idx, "is_correct"] == 1:
            correct_top1 += 1
            
    cond_top1 = correct_top1 / gt_in_pool if gt_in_pool > 0 else 0
    return cond_top1, gt_in_pool, correct_top1

def ranker_v18c_original(pool):
    scores = []
    # simulate original fam_pop directly
    for _, c in pool.iterrows():
        fam_pop = c["family_population"]  # original count
        ctx = c["context_128"] if not pd.isna(c["context_128"]) else 0.0
        phase_pen = c["phase_penalty"] if not pd.isna(c["phase_penalty"]) else 0.0
        w_center = 0.12 if fam_pop > 3 else 0.04
        center_penalty = (c["dist_to_center"] / 250.0) ** 2
        score = c["corr_score"] + 0.15 * ctx - 0.20 * phase_pen - w_center * center_penalty
        scores.append(score)
    return scores

def ranker_v24a_invariant(pool):
    scores = []
    pool_size = len(pool)
    for _, c in pool.iterrows():
        # normalized family density (using the true pool-wide density, or simulated)
        # Note: 'family_population' in csv was calculated at K=200.
        # We need to approximate what it would be at K.
        # For simplicity, if we assume family members are evenly distributed, 
        # family_ratio = (fam_pop_at_200 / 200).
        family_ratio = c["family_population"] / 200.0 
        
        ctx = c["context_128"] if not pd.isna(c["context_128"]) else 0.0
        phase_pen = c["phase_penalty"] if not pd.isna(c["phase_penalty"]) else 0.0
        
        # 4/50 = 0.08
        w_center = 0.12 if family_ratio > 0.08 else 0.04
        center_penalty = (c["dist_to_center"] / 250.0) ** 2
        score = c["corr_score"] + 0.15 * ctx - 0.20 * phase_pen - w_center * center_penalty
        scores.append(score)
    return scores

def ranker_v24b_adaptive(pool):
    scores = []
    pool_size = len(pool)
    # Compute image-level periodicity strength (e.g., max family ratio)
    max_family_ratio = (pool["family_population"].max() / 200.0)
    
    for _, c in pool.iterrows():
        ctx = c["context_128"] if not pd.isna(c["context_128"]) else 0.0
        phase_pen = c["phase_penalty"] if not pd.isna(c["phase_penalty"]) else 0.0
        
        # Adaptive center weight based on overall image periodicity
        # If max_family_ratio > 0.20 (20% of peaks are in one family), heavy periodicity -> heavy center weight
        base_w = 0.02
        periodicity_factor = np.clip((max_family_ratio - 0.05) / 0.15, 0.0, 1.0) # 0 to 1
        w_center = base_w + 0.15 * periodicity_factor
        
        center_penalty = (c["dist_to_center"] / 250.0) ** 2
        score = c["corr_score"] + 0.15 * ctx - 0.20 * phase_pen - w_center * center_penalty
        scores.append(score)
    return scores

if __name__ == "__main__":
    df = pd.read_csv("phase2/V22_CHAMPIONSHIP/results/candidate_pool_features.csv")
    print(f"Loaded {len(df)} candidates for {df.pair_id.nunique()} pairs.")
    
    for K in [50, 100, 200]:
        print(f"\\n=== K = {K} ===")
        
        c1, g1, r1 = eval_ranker(df, ranker_v18c_original, K)
        print(f"V18-C Original: {c1*100:.2f}% ({r1}/{g1})")
        
        c2, g2, r2 = eval_ranker(df, ranker_v24a_invariant, K)
        print(f"V24-A Invariant: {c2*100:.2f}% ({r2}/{g2})")
        
        c3, g3, r3 = eval_ranker(df, ranker_v24b_adaptive, K)
        print(f"V24-B Adaptive: {c3*100:.2f}% ({r3}/{g3})")

