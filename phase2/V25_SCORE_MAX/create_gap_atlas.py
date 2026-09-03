import pandas as pd
import numpy as np
import pickle
import json
import os

def create_gap_atlas():
    out_dir = 'phase2/V25_SCORE_MAX'
    os.makedirs(out_dir, exist_ok=True)
    
    pairs = pd.read_csv('data/phase2_dev/pairs.csv')
    df = pd.read_csv('phase2/V25_CHAMPIONSHIP/v25_train_features.csv')
    
    with open('phase2/V25_CHAMPIONSHIP/ranker.pkl', 'rb') as f:
        ranker = pickle.load(f)
        
    # Relative features for ranker
    feature_cols = ['corr_score', 'psr', 'context_128', 'context_combined', 'phase_penalty', 
                   'dist_to_center', 'neigh_cons', 'grad_ncc']
    for c in feature_cols:
        df[c + '_rel'] = df.groupby('pair_id')[c].transform(lambda x: x - x.median())
    df['family_ratio'] = df.groupby('pair_id')['family_population'].transform(lambda x: x / x.count())
    
    X_cols = ranker['features']
    df['ranker_score'] = ranker['model'].predict_proba(df[X_cols])[:, 1]
    
    # -------------------------------------------------------------------------
    # 1. LOAD V25 CANDIDATE POOL (sort descending)
    # -------------------------------------------------------------------------
    df = df.sort_values(['pair_id', 'ranker_score'], ascending=[True, False]).reset_index(drop=True)
    # Add a formal rank_idx (1-indexed for the gaps logic)
    df['v25_rank'] = df.groupby('pair_id').cumcount() + 1

    gap_atlas_rows = []
    gt_gaps_rows = []
    pairwise_rows = []
    failure_groups = {} # pid -> group
    rescue_rows = []

    for pair_id, group in df.groupby('pair_id'):
        pair_info = pairs[pairs['pair_id'] == pair_id].iloc[0]
        set_type = pair_info['set_type']
        is_present = 1 if set_type in ['SetA', 'SetB'] else 0
        
        scores = group['ranker_score'].values
        ranks = group['v25_rank'].values
        labels = group['label'].values
        
        score_1 = scores[0]
        
        # -------------------------------------------------------------------------
        # 2. GLOBAL RANK GAPS
        # -------------------------------------------------------------------------
        gap_dict = {'pair_id': pair_id}
        for k in [2, 3, 5, 10, 20, 30, 50, 100, 200]:
            if len(scores) >= k:
                gap = score_1 - scores[k-1]
                gap_dict[f'gap_1_{k}'] = gap
                if k in [2, 5, 10, 20]:
                    gap_dict[f'relative_gap_1_{k}'] = gap / (abs(score_1) + 1e-6)
            else:
                gap_dict[f'gap_1_{k}'] = np.nan
                if k in [2, 5, 10, 20]:
                    gap_dict[f'relative_gap_1_{k}'] = np.nan
                    
        gap_atlas_rows.append(gap_dict)
        
        # -------------------------------------------------------------------------
        # 3. GT-RELATIVE GAPS & 5. FAILURE GROUPS & 8. RESCUE POTENTIAL
        # -------------------------------------------------------------------------
        group_num = 7 # default GT outside top-200 or absent
        if is_present:
            gt_indices = np.where(labels == 1)[0]
            if len(gt_indices) > 0:
                # take highest-ranked GT if multiple exist
                gt_idx = gt_indices[0] 
                gt_rank = ranks[gt_idx]
                gt_score = scores[gt_idx]
                winner_score = score_1
                
                gt_minus_winner = gt_score - winner_score
                winner_minus_gt = winner_score - gt_score
                
                def safe_gap(g_score, scores_arr, k):
                    if len(scores_arr) >= k: return g_score - scores_arr[k-1]
                    return np.nan
                
                gt_gaps_rows.append({
                    'pair_id': pair_id,
                    'set_type': set_type,
                    'gt_rank': gt_rank,
                    'gt_score': gt_score,
                    'winner_score': winner_score,
                    'gt_minus_winner': gt_minus_winner,
                    'gt_minus_top2': safe_gap(gt_score, scores, 2),
                    'gt_minus_top5': safe_gap(gt_score, scores, 5),
                    'gt_minus_top10': safe_gap(gt_score, scores, 10),
                    'gt_minus_top20': safe_gap(gt_score, scores, 20),
                    'winner_minus_gt': winner_minus_gt
                })
                
                rescue_rows.append({
                    'pair_id': pair_id,
                    'required_ranker_delta': winner_minus_gt
                })
                
                if gt_rank == 1: group_num = 1
                elif gt_rank <= 10: group_num = 2
                elif gt_rank <= 20: group_num = 3
                elif gt_rank <= 50: group_num = 4
                elif gt_rank <= 100: group_num = 5
                elif gt_rank <= 200: group_num = 6
            else:
                group_num = 7
                
            failure_groups[pair_id] = group_num
            
        # -------------------------------------------------------------------------
        # 4. CANDIDATE PAIR GAPS
        # -------------------------------------------------------------------------
        # Create pairs: Top-1 vs Top-2..20, GT vs Top-1..20
        # To avoid combinatorial explosion, we only generate a subset of interesting pairs
        if is_present and group_num != 7: # If GT is in Top-200
            gt_row = group[group['label'] == 1].iloc[0]
            # Pair GT with wrong Top candidates
            for idx, cand_row in group.head(20).iterrows():
                if cand_row['label'] == 1: continue # Don't pair GT with itself
                
                row_dict = {
                    'pair_id': pair_id,
                    'candidate_a': 'GT',
                    'candidate_b': f'Rank_{cand_row["v25_rank"]}',
                    'rank_a': gt_row['v25_rank'],
                    'rank_b': cand_row['v25_rank'],
                    'score_a': gt_row['ranker_score'],
                    'score_b': cand_row['ranker_score'],
                    'delta_score': gt_row['ranker_score'] - cand_row['ranker_score'],
                    'delta_corr': gt_row['corr_score'] - cand_row['corr_score'],
                    'delta_psr': gt_row['psr'] - cand_row['psr'],
                    'delta_context128': gt_row['context_128'] - cand_row['context_128'],
                    'delta_context_combined': gt_row['context_combined'] - cand_row['context_combined'],
                    'delta_neigh': gt_row['neigh_cons'] - cand_row['neigh_cons'],
                    'delta_gradient': gt_row['grad_ncc'] - cand_row['grad_ncc'],
                    'delta_phase': gt_row['phase_penalty'] - cand_row['phase_penalty'],
                    'delta_center': gt_row['dist_to_center'] - cand_row['dist_to_center'],
                    'delta_family': gt_row['family_ratio'] - cand_row['family_ratio'],
                    'label': 1
                }
                pairwise_rows.append(row_dict)

    # Convert to DataFrames
    df_gap = pd.DataFrame(gap_atlas_rows)
    df_gt = pd.DataFrame(gt_gaps_rows)
    df_pw = pd.DataFrame(pairwise_rows)
    df_res = pd.DataFrame(rescue_rows).sort_values('required_ranker_delta', ascending=True)
    
    # Save CSVs
    df_gap.to_csv(f'{out_dir}/gap_atlas.csv', index=False)
    if len(df_gt) > 0: df_gt.to_csv(f'{out_dir}/gt_relative_gaps.csv', index=False)
    if len(df_pw) > 0: df_pw.to_csv(f'{out_dir}/pairwise_gap_features.csv', index=False)
    if len(df_res) > 0: df_res.to_csv(f'{out_dir}/rescue_potential.csv', index=False)
    
    # -------------------------------------------------------------------------
    # 6 & 7. GAP STATISTICS (Global and Structural)
    # -------------------------------------------------------------------------
    stats_dict = {}
    
    for g in range(1, 8):
        pids = [k for k, v in failure_groups.items() if v == g]
        stats_dict[f'GROUP_{g}'] = {'count': len(pids)}
        if len(pids) == 0: continue
        
        # Calculate stats on gaps
        group_df_gap = df_gap[df_gap['pair_id'].isin(pids)]
        for c in ['gap_1_2', 'gap_1_5', 'gap_1_10', 'gap_1_20']:
            if c in group_df_gap.columns:
                vals = group_df_gap[c].dropna().values
                if len(vals) > 0:
                    stats_dict[f'GROUP_{g}'][c] = {
                        'mean': np.mean(vals), 'median': np.median(vals), 'std': np.std(vals),
                        'min': np.min(vals), 'max': np.max(vals),
                        'p05': np.percentile(vals, 5), 'p10': np.percentile(vals, 10),
                        'p25': np.percentile(vals, 25), 'p50': np.percentile(vals, 50),
                        'p75': np.percentile(vals, 75), 'p90': np.percentile(vals, 90),
                        'p95': np.percentile(vals, 95)
                    }
        
        # GT minus winner
        group_df_gt = df_gt[df_gt['pair_id'].isin(pids)]
        if len(group_df_gt) > 0:
            vals = group_df_gt['gt_minus_winner'].dropna().values
            if len(vals) > 0:
                stats_dict[f'GROUP_{g}']['gt_minus_winner'] = {
                    'mean': np.mean(vals), 'median': np.median(vals), 'std': np.std(vals),
                    'min': np.min(vals), 'max': np.max(vals),
                    'p05': np.percentile(vals, 5), 'p10': np.percentile(vals, 10),
                    'p25': np.percentile(vals, 25), 'p50': np.percentile(vals, 50),
                    'p75': np.percentile(vals, 75), 'p90': np.percentile(vals, 90),
                    'p95': np.percentile(vals, 95)
                }
                
        # structural gaps (winner vs GT)
        # We can extract this from pairwise_gap_features where label=1 (meaning GT vs someone)
        # and candidate_b == 'Rank_1'
        group_df_pw = df_pw[(df_pw['pair_id'].isin(pids)) & (df_pw['candidate_b'] == 'Rank_1')]
        structural_cols = ['delta_corr', 'delta_psr', 'delta_context_combined', 'delta_neigh', 'delta_gradient', 'delta_phase', 'delta_center', 'delta_family']
        for c in structural_cols:
            if c in group_df_pw.columns:
                vals = group_df_pw[c].dropna().values
                if len(vals) > 0:
                    stats_dict[f'GROUP_{g}'][c] = {
                        'mean': np.mean(vals), 'median': np.median(vals), 'std': np.std(vals),
                        'min': np.min(vals), 'max': np.max(vals),
                        'p05': np.percentile(vals, 5), 'p10': np.percentile(vals, 10),
                        'p25': np.percentile(vals, 25), 'p50': np.percentile(vals, 50),
                        'p75': np.percentile(vals, 75), 'p90': np.percentile(vals, 90),
                        'p95': np.percentile(vals, 95)
                    }

    with open(f'{out_dir}/gap_statistics.json', 'w') as f:
        json.dump(stats_dict, f, indent=4)
        
    # -------------------------------------------------------------------------
    # 10. FINAL REPORT
    # -------------------------------------------------------------------------
    counts = {g: len([k for k, v in failure_groups.items() if v == g]) for g in range(1, 8)}
    
    print('TOTAL PRESENT', sum(counts.values()))
    print('GT IN TOP10  ', counts.get(1,0) + counts.get(2,0))
    print('GT IN TOP20  ', counts.get(1,0) + counts.get(2,0) + counts.get(3,0))
    print('GT IN TOP50  ', counts.get(1,0) + counts.get(2,0) + counts.get(3,0) + counts.get(4,0))
    print('GT IN TOP100 ', counts.get(1,0) + counts.get(2,0) + counts.get(3,0) + counts.get(4,0) + counts.get(5,0))
    print('GT IN TOP200 ', sum([counts.get(g,0) for g in range(1,7)]))
    print('GT OUTSIDE TOP200', counts.get(7,0))
    
    print('\nMedian winner-GT score gap:')
    if len(df_gt) > 0:
        gaps = df_gt[df_gt['gt_rank'] > 1]['winner_minus_gt'].dropna().values
        if len(gaps) > 0:
            print(f"P25: {np.percentile(gaps, 25):.4f}")
            print(f"P50: {np.percentile(gaps, 50):.4f}")
            print(f"P75: {np.percentile(gaps, 75):.4f}")
            print(f"P90: {np.percentile(gaps, 90):.4f}")
            print(f"P95: {np.percentile(gaps, 95):.4f}")
        else:
            print("No misranked GT cases to calculate gap on.")
            
    print('\nHow many GT candidates require:')
    if len(df_res) > 0:
        req = df_res[df_res['required_ranker_delta'] > 0]['required_ranker_delta'].values
        print(f"<0.005: {len(req[req < 0.005])}")
        print(f"<0.010: {len(req[req < 0.010])}")
        print(f"<0.020: {len(req[req < 0.020])}")
        print(f"<0.050: {len(req[req < 0.050])}")
        print(f"<0.100: {len(req[req < 0.100])}")
        print(f">=0.100: {len(req[req >= 0.100])}")
        
        # Determine gap discriminability
        # If a large portion requires >= 0.050, it's non-discriminative
        if len(req[req >= 0.050]) / len(req) > 0.5:
            print('\nCONCLUSION: C) gap mostly non-discriminative (large gaps dominate)')
        elif len(req[req < 0.010]) / len(req) > 0.5:
            print('\nCONCLUSION: A) gap appears highly discriminative (small gaps dominate)')
        else:
            print('\nCONCLUSION: B) gap partially discriminative')
            

if __name__ == '__main__':
    create_gap_atlas()
