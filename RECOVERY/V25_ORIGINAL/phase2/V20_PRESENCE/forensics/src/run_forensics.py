import os
import sys
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from unittest.mock import patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from dataset_generator import generate_finfet_layout, generate_pair
from phase2.inference_phase2 import perform_phase2_localization
from phase2 import candidate_ranker

def generate_v20_datasets():
    canvas = generate_finfet_layout(10000, 10000, seed=42)
    set_a, set_b, set_c = [], [], []
    for i in range(40):
        ref, search, x_t, y_t, s_t, r_t, nl = generate_pair(canvas, f"A_{i}", "finfet", "val", 1.0, seed=42+i)
        set_a.append({"pair_id": f"A_{i}", "ref": ref, "search": search, "gt_found": 1, "set": "A_NOMINAL"})
    for i in range(28):
        ref, search, x_t, y_t, s_t, r_t, nl = generate_pair(canvas, f"B_{i}", "finfet", "val", 2.5, seed=100+i)
        set_b.append({"pair_id": f"B_{i}", "ref": ref, "search": search, "gt_found": 1, "set": "B_DEGRADED"})
    for i in range(22):
        ref, search, x_t, y_t, s_t, r_t, nl = generate_pair(canvas, f"C_{i}", "finfet", "val", 1.0, seed=200+i)
        h, w = canvas.shape
        search_size = 1000
        sw_patch = int(round(search_size / 0.10))
        sh_patch = int(round(search_size / 0.10))
        sx1 = min(w - sw_patch, 0)
        sy1 = min(h - sh_patch, 0)
        search_raw = canvas[sy1:sy1+sh_patch, sx1:sx1+sw_patch]
        search_img_resized = cv2.resize(search_raw, (search_size, search_size), interpolation=cv2.INTER_AREA)
        from dataset_generator import apply_sem_acquisition_effects
        search_final = apply_sem_acquisition_effects(
            search_img_resized, blur_sigma=1.0, dose_lambda=80.0,
            gaussian_noise_std=0.03, edge_factor=0.12, charging_std=0.02, seed=200+i
        )
        set_c.append({"pair_id": f"C_{i}", "ref": ref, "search": search_final, "gt_found": 0, "set": "C_ABSENT"})
    return set_a + set_b + set_c

all_candidates_data = []

original_rank_candidates = candidate_ranker.rank_candidates
def patched_rank_candidates(candidates):
    ranked = original_rank_candidates(candidates)
    if hasattr(patched_rank_candidates, "current_pair"):
        for c in ranked:
            cd = c.copy()
            cd["pair_id"] = patched_rank_candidates.current_pair["pair_id"]
            cd["set"] = patched_rank_candidates.current_pair["set"]
            cd["gt_found"] = patched_rank_candidates.current_pair["gt_found"]
            cd["total_candidates"] = len(ranked)
            all_candidates_data.append(cd)
    return ranked

def main():
    datasets = generate_v20_datasets()
    print(f"Extracted datasets: {len(datasets)}")
    
    with patch("phase2.inference_phase2.rank_candidates", side_effect=patched_rank_candidates):
        for item in datasets:
            patched_rank_candidates.current_pair = item
            perform_phase2_localization(item["ref"], item["search"])
            
    df = pd.DataFrame(all_candidates_data)
    
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, 'feature_distribution.csv'), index=False)
    
    # We want to distinguish Set A/B (gt_found=1) vs Set C (gt_found=0)
    # The question is: "What physical/evidence characteristics distinguish a genuinely present structure from a hard negative that produces a convincing correlation peak?"
    
    # Take the top candidate for each pair (candidate with rank_score highest)
    df_top = df.loc[df.groupby('pair_id')['rank_score'].idxmax()].copy()
    
    df_top.to_csv(os.path.join(out_dir, 'false_positive_case_table.csv'), index=False)
    
    metrics = ['corr_score', 'psr', 'peak_margin', 'context_128', 'phase_residual', 
               'family_population', 'local_density', 'center_prior', 'nearest_edge_dist', 'nearest_cut_dist']
               
    stats_list = []
    for m in metrics:
        if m not in df_top.columns:
            continue
        nom = df_top[df_top['set'] == 'A_NOMINAL'][m]
        deg = df_top[df_top['set'] == 'B_DEGRADED'][m]
        absent = df_top[df_top['set'] == 'C_ABSENT'][m]
        
        stats_list.append({
            'Feature': m,
            'Nominal_Mean': nom.mean(),
            'Nominal_Median': nom.median(),
            'Degraded_Mean': deg.mean(),
            'Degraded_Median': deg.median(),
            'Absent_Mean': absent.mean(),
            'Absent_Median': absent.median()
        })
    stats_df = pd.DataFrame(stats_list)
    stats_df.to_csv(os.path.join(out_dir, 'pairwise_feature_statistics.csv'), index=False)
    
    # Categorization of FP cases
    c_cases = df_top[df_top['set'] == 'C_ABSENT'].copy()
    mechanisms = []
    for _, row in c_cases.iterrows():
        # Heuristics for failure mechanism
        if row.get('family_population', 1) > 2:
            mechanisms.append('PERIODIC_REPLICA')
        elif row.get('context_128', 1.0) < 0.4:
            mechanisms.append('CONTEXT_MISMATCH')
        elif row.get('phase_residual', 0) > 0.5:
            mechanisms.append('PHASE_INCONSISTENCY')
        elif row.get('peak_margin', 1.0) < 0.1:
            mechanisms.append('MULTI_PEAK_AMBIGUITY')
        elif row.get('nearest_edge_dist', 100) < 50:
            mechanisms.append('BOUNDARY_MATCH')
        else:
            mechanisms.append('GLOBAL_TEXTURE_MATCH')
            
    c_cases['Failure_Mechanism'] = mechanisms
    c_cases[['pair_id', 'Failure_Mechanism']].to_csv(os.path.join(out_dir, 'mechanism_categorization.csv'), index=False)
    
    # Plots
    plot_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'plots'))
    os.makedirs(plot_dir, exist_ok=True)
    
    for m in metrics:
        if m in df_top.columns:
            plt.figure()
            sns.boxplot(x='set', y=m, data=df_top)
            plt.title(f'Distribution of {m}')
            plt.savefig(os.path.join(plot_dir, f'{m}_boxplot.png'))
            plt.close()
            
    print("Forensics complete.")

if __name__ == "__main__":
    main()
