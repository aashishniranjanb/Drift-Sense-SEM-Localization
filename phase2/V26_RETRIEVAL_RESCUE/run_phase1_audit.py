import os
import sys
import cv2
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('phase2')
sys.path.append('fallbacks')
sys.path.append('team/akhilesh-localization')
sys.path.append('production_engine')

from pose_fallback import perform_pose_fallback_search
from candidate_extractor import extract_nms_fast, extract_candidates_akhilesh
from inference_phase2 import cluster_replica_families, verify_candidate_context, verify_phase_consistency
from phase2.V25_CHAMPIONSHIP.periodicity import estimate_periodicity_from_corr
from phase2.V25_CHAMPIONSHIP.feature_extractors import compute_neighborhood_consistency, compute_gradient_ncc

pairs = pd.read_csv('data/phase2_dev/pairs.csv')
v25_tax = pd.read_csv('data/phase2_dev/failure_taxonomy.csv')
v25_features = pd.read_csv('phase2/V25_CHAMPIONSHIP/v25_train_features.csv')

# Load the 35 outside-200 pair IDs
df_gt = pd.read_csv('phase2/V25_SCORE_MAX/gt_relative_gaps.csv')
present_pairs = pairs[pairs['set_type'].isin(['SetA', 'SetB'])]['pair_id'].tolist()
in_200 = set(df_gt['pair_id'])
out_200_ids = set([p for p in present_pairs if p not in in_200])

print('Total outside 200:', len(out_200_ids))

out_rows = []
tax_rows = []
ceiling_counts = {
    'total_present': len(present_pairs),
    'gt_in_top200': len(in_200),
    'gt_in_top500': 0,
    'gt_in_top1000': 0,
    'gt_raw_candidate_coverage': 0
}

alt_signals_ranks = {
    'ncc': [],
    'psr': [],
    'context': [],
    'gradient': [],
    'phase': [],
    'center_dist': []
}

for pid in tqdm(out_200_ids):
    row = pairs[pairs['pair_id'] == pid].iloc[0]
    gt_x = float(row['gt_x'])
    gt_y = float(row['gt_y'])
    set_type = row['set_type']
    
    ref_path = os.path.join('data/phase2_dev', row['reference_path'])
    search_path = os.path.join('data/phase2_dev', row['search_path'])
    
    ref = cv2.imread(ref_path, 0)
    search = cv2.imread(search_path, 0)
    
    pose = perform_pose_fallback_search(ref, search)
    corr_plane = pose['corr_plane']
    best_template = pose['best_template']
    est_scale = pose['best_scale']
    est_theta = pose['best_theta']
    tw, th = best_template.shape[::-1]
    
    sh, sw = search.shape[:2]
    search_cx, search_cy = sw / 2.0, sh / 2.0
    gt_dist_center = np.hypot(gt_x - search_cx, gt_y - search_cy)
    
    periodicity = estimate_periodicity_from_corr(corr_plane)
    pitch_x = periodicity['pitch_x']
    pitch_y = periodicity['pitch_y']
    
    p_feats = v25_features[v25_features['pair_id'] == pid]
    best_v25_row = p_feats.iloc[0] if len(p_feats) > 0 else None
    
    v25_cands = extract_candidates_akhilesh(corr_plane, tw, th, ref, search, est_scale, est_theta, max_final_k=200)
    best_cand = v25_cands[0]
    
    dists_to_gt = [np.hypot(c['cx'] - gt_x, c['cy'] - gt_y) for c in v25_cands]
    min_dist_to_gt = min(dists_to_gt) if len(dists_to_gt) > 0 else -1.0
    
    gt_px = int(round(gt_x - tw / 2.0))
    gt_py = int(round(gt_y - th / 2.0))
    
    gt_px_clamped = max(0, min(corr_plane.shape[1]-1, gt_px))
    gt_py_clamped = max(0, min(corr_plane.shape[0]-1, gt_py))
    gt_local_ncc = float(corr_plane[gt_py_clamped, gt_px_clamped])
    
    y1, y2 = max(0, gt_py_clamped - 15), min(corr_plane.shape[0], gt_py_clamped + 16)
    x1, x2 = max(0, gt_px_clamped - 15), min(corr_plane.shape[1], gt_px_clamped + 16)
    patch = corr_plane[y1:y2, x1:x2]
    mean_val = np.mean(patch)
    std_val = np.std(patch)
    gt_psr = float((gt_local_ncc - mean_val) / (std_val + 1e-6))
    
    gt_ctx = verify_candidate_context(ref, search, gt_x, gt_y, est_scale, est_theta)
    gt_phase = verify_phase_consistency(search, best_template, gt_px_clamped, gt_py_clamped)
    gt_neigh = compute_neighborhood_consistency(search, best_template, gt_px_clamped, gt_py_clamped, pitch_x, pitch_y)
    gt_grad = compute_gradient_ncc(search, best_template, gt_px_clamped, gt_py_clamped)
    
    out_rows.append({
        'pair_id': pid,
        'set_type': set_type,
        'gt_x': gt_x,
        'gt_y': gt_y,
        'gt_dist_center': gt_dist_center,
        'nearest_v25_dist_to_gt': min_dist_to_gt,
        'gt_local_ncc': gt_local_ncc,
        'gt_psr': gt_psr,
        'gt_context32': gt_ctx['s32'],
        'gt_context64': gt_ctx['s64'],
        'gt_context128': gt_ctx['s128'],
        'gt_context_combined': gt_ctx['combined'],
        'gt_gradient_ncc': gt_grad,
        'gt_phase_score': gt_phase,
        'gt_neigh_score': gt_neigh,
        'pitch_x': pitch_x,
        'pitch_y': pitch_y,
        'best_v25_x': best_cand['cx'],
        'best_v25_y': best_cand['cy'],
        'best_v25_score': best_cand['corr_score'],
        'best_v25_dist_to_center': np.hypot(best_cand['cx'] - search_cx, best_cand['cy'] - search_cy)
    })
    
    raw_1000 = extract_nms_fast(corr_plane, tw, th, max_k=1000, r=5)
    
    sub_y1, sub_y2 = max(0, gt_py_clamped - 5), min(corr_plane.shape[0], gt_py_clamped + 6)
    sub_x1, sub_x2 = max(0, gt_px_clamped - 5), min(corr_plane.shape[1], gt_px_clamped + 6)
    sub_corr = corr_plane[sub_y1:sub_y2, sub_x1:sub_x2]
    local_peak_exists = bool(np.max(sub_corr) == corr_plane[gt_py_clamped, gt_px_clamped] or (gt_local_ncc > 0.15))
    
    gt_raw_rank = None
    for idx_c, c in enumerate(raw_1000):
        if np.hypot(c['cx'] - gt_x, c['cy'] - gt_y) <= 5.0:
            gt_raw_rank = idx_c + 1
            break
            
    gt_in_raw = gt_raw_rank is not None
    if not gt_in_raw:
        raw_all = extract_nms_fast(corr_plane, tw, th, max_k=3000, r=5)
        for idx_c, c in enumerate(raw_all):
            if np.hypot(c['cx'] - gt_x, c['cy'] - gt_y) <= 5.0:
                gt_raw_rank = idx_c + 1
                gt_in_raw = True
                break
                
    tax_cat = 'UNKNOWN'
    if gt_dist_center > 260.0 and gt_raw_rank is not None and gt_raw_rank <= 200:
        tax_cat = 'CENTER_BIAS'
    elif gt_local_ncc < 0.20:
        if set_type == 'SetB':
            tax_cat = 'DEGRADATION'
        else:
            tax_cat = 'SCALE_ROTATION'
    elif periodicity['mode'] == 'STRONG' and min_dist_to_gt > 15.0 and min_dist_to_gt < 120.0:
        tax_cat = 'PERIODIC'
    elif gt_raw_rank is not None and gt_raw_rank > 200 and gt_raw_rank <= 1000:
        tax_cat = 'SPATIAL'
    elif not local_peak_exists or (gt_raw_rank is not None and gt_raw_rank > 1000):
        tax_cat = 'LOW_SIGNAL'
    elif min_dist_to_gt <= 10.0 and min_dist_to_gt > 5.0:
        tax_cat = 'NMS_SUPPRESSION'
    else:
        if set_type == 'SetB':
            tax_cat = 'DEGRADATION'
        else:
            tax_cat = 'SPATIAL'
            
    tax_rows.append({
        'pair_id': pid,
        'set_type': set_type,
        'gt_raw_rank': gt_raw_rank if gt_raw_rank is not None else -1,
        'local_peak_exists': local_peak_exists,
        'gt_dist_center': gt_dist_center,
        'min_dist_to_gt': min_dist_to_gt,
        'gt_local_ncc': gt_local_ncc,
        'taxonomy_category': tax_cat
    })
    
    eval_pool = raw_1000[:500]
    gt_in_eval_idx = None
    for idx_c, c in enumerate(eval_pool):
        if np.hypot(c['cx'] - gt_x, c['cy'] - gt_y) <= 5.0:
            gt_in_eval_idx = idx_c
            break
            
    if gt_in_eval_idx is not None:
        cand_ncc = [c['corr_score'] for c in eval_pool]
        cand_center = [-np.hypot(c['cx'] - search_cx, c['cy'] - search_cy) for c in eval_pool]
        
        rank_ncc = 1 + sum(1 for s in cand_ncc if s > cand_ncc[gt_in_eval_idx])
        alt_signals_ranks['ncc'].append(rank_ncc)
        
        rank_center = 1 + sum(1 for s in cand_center if s > cand_center[gt_in_eval_idx])
        alt_signals_ranks['center_dist'].append(rank_center)
        
        c_ctxs = []
        c_grads = []
        c_phases = []
        for c in eval_pool[:60]:
            ctx = verify_candidate_context(ref, search, c['cx'], c['cy'], est_scale, est_theta)
            grad = compute_gradient_ncc(search, best_template, c['peak_x'], c['peak_y'])
            ph = verify_phase_consistency(search, best_template, c['peak_x'], c['peak_y'])
            c_ctxs.append(ctx['combined'])
            c_grads.append(grad)
            c_phases.append(-ph)
            
        if gt_in_eval_idx < 60:
            rank_ctx = 1 + sum(1 for s in c_ctxs if s > c_ctxs[gt_in_eval_idx])
            rank_grad = 1 + sum(1 for s in c_grads if s > c_grads[gt_in_eval_idx])
            rank_phase = 1 + sum(1 for s in c_phases if s > c_phases[gt_in_eval_idx])
            alt_signals_ranks['context'].append(rank_ctx)
            alt_signals_ranks['gradient'].append(rank_grad)
            alt_signals_ranks['phase'].append(rank_phase)

print('\nComputing overall ceiling on all 140 present pairs...')
gt_top500_total = len(in_200)
gt_top1000_total = len(in_200)
gt_raw_total = len(in_200)

for r in tax_rows:
    rk = r['gt_raw_rank']
    if rk > 0:
        gt_raw_total += 1
        if rk <= 500:
            gt_top500_total += 1
        if rk <= 1000:
            gt_top1000_total += 1

ceiling_counts['gt_in_top500'] = gt_top500_total
ceiling_counts['gt_in_top1000'] = gt_top1000_total
ceiling_counts['gt_raw_candidate_coverage'] = gt_raw_total

df_outside = pd.DataFrame(out_rows)
df_outside.to_csv('phase2/V26_RETRIEVAL_RESCUE/outside200.csv', index=False)

df_taxonomy = pd.DataFrame(tax_rows)
df_taxonomy.to_csv('phase2/V26_RETRIEVAL_RESCUE/retrieval_failure_taxonomy.csv', index=False)

sig_cov_rows = []
for sig_name, rks in alt_signals_ranks.items():
    if len(rks) == 0:
        continue
    rks_arr = np.array(rks)
    sig_cov_rows.append({
        'signal': sig_name,
        'evaluated_cases': len(rks_arr),
        'in_top10': int(np.sum(rks_arr <= 10)),
        'in_top20': int(np.sum(rks_arr <= 20)),
        'in_top50': int(np.sum(rks_arr <= 50)),
        'in_top100': int(np.sum(rks_arr <= 100)),
        'in_top200': int(np.sum(rks_arr <= 200))
    })
df_sig = pd.DataFrame(sig_cov_rows)
df_sig.to_csv('phase2/V26_RETRIEVAL_RESCUE/retrieval_signal_coverage.csv', index=False)

with open('phase2/V26_RETRIEVAL_RESCUE/retrieval_ceiling.json', 'w') as f:
    json.dump(ceiling_counts, f, indent=4)

print('\n=============================================')
print('V26 RETRIEVAL RESCUE - PHASE 1 RESULTS')
print('=============================================')
tot = ceiling_counts['total_present']
top200 = ceiling_counts['gt_in_top200']
top500 = ceiling_counts['gt_in_top500']
top1000 = ceiling_counts['gt_in_top1000']
raw_tot = ceiling_counts['gt_raw_candidate_coverage']

print('Total Present Pairs:          ' + str(tot))
print('GT in Top-200 (V25 Baseline): ' + str(top200) + ' / ' + str(tot) + ' (' + str(round(top200/tot*100, 1)) + '%)')
print('GT in Top-500:                ' + str(top500) + ' / ' + str(tot) + ' (' + str(round(top500/tot*100, 1)) + '%)')
print('GT in Top-1000:               ' + str(top1000) + ' / ' + str(tot) + ' (' + str(round(top1000/tot*100, 1)) + '%)')
print('GT Raw Candidate Coverage:    ' + str(raw_tot) + ' / ' + str(tot) + ' (' + str(round(raw_tot/tot*100, 1)) + '%)')

print('\nFailure Counts by Category (35 Outside-200 Cases):')
cat_counts = df_taxonomy['taxonomy_category'].value_counts()
for cat, cnt in cat_counts.items():
    print('  ' + f'{cat:20s}' + ': ' + f'{cnt:2d}' + ' (' + str(round(cnt/35.0*100, 1)) + '%)')
