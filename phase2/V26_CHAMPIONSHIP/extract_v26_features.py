import sys
import os
import cv2
import numpy as np
import pandas as pd
import pickle
import time
from tqdm import tqdm

sys.path.append('phase2')
sys.path.append('fallbacks')
sys.path.append('team/akhilesh-localization')
sys.path.append('production_engine')

from pose_fallback import perform_pose_fallback_search
from candidate_extractor import extract_candidates_akhilesh, extract_nms_fast
from inference_phase2 import cluster_replica_families, verify_candidate_context, verify_phase_consistency
from phase2.V25_CHAMPIONSHIP.periodicity import estimate_periodicity_from_corr
from phase2.V25_CHAMPIONSHIP.feature_extractors import compute_neighborhood_consistency, compute_gradient_ncc

def extract_v26_features():
    pairs = pd.read_csv('data/phase2_dev/pairs.csv')
    
    with open('phase2/V25_CHAMPIONSHIP/ranker.pkl', 'rb') as f:
        ranker_model = pickle.load(f)
        
    all_features = []
    
    for idx, row in tqdm(pairs.iterrows(), total=len(pairs)):
        pid = row['pair_id']
        ref = cv2.imread(os.path.join('data/phase2_dev', row['reference_path']), 0)
        search = cv2.imread(os.path.join('data/phase2_dev', row['search_path']), 0)
        
        # 1. Pose Fallback (Base Correlator)
        pose = perform_pose_fallback_search(ref, search)
        corr_plane = pose['corr_plane']
        best_template = pose['best_template']
        est_scale = pose['best_scale']
        est_theta = pose['best_theta']
        tw, th = best_template.shape[::-1]
        
        sh, sw = search.shape[:2]
        search_cx, search_cy = sw / 2.0, sh / 2.0
        
        # 2. Extract Queues
        cands_v25 = extract_candidates_akhilesh(corr_plane, tw, th, ref, search, est_scale, est_theta, max_final_k=200)
        for c in cands_v25: c['queue'] = 'V25'
        
        cands_r3 = extract_nms_fast(corr_plane, tw, th, max_k=150, r=3)
        for c in cands_r3: c['queue'] = 'R3'
            
        cands_r2 = extract_nms_fast(corr_plane, tw, th, max_k=150, r=2)
        for c in cands_r2: c['queue'] = 'R2'
            
        # 3. Union and Deduplicate
        final_pool = list(cands_v25)
        rescue_pool = []
        
        def is_duplicate(c, pool, threshold=2.0):
            for p in pool:
                if np.hypot(c['cx'] - p['cx'], c['cy'] - p['cy']) <= threshold:
                    return True
            return False
            
        for c in cands_r3 + cands_r2:
            c['dist_to_center'] = float(np.hypot(c['cx'] - search_cx, c['cy'] - search_cy))
            if not is_duplicate(c, final_pool) and not is_duplicate(c, rescue_pool):
                rescue_pool.append(c)
                
        # Keep top rescues by corr_score to limit expensive features
        rescue_pool.sort(key=lambda x: x['corr_score'], reverse=True)
        rescue_pool = rescue_pool[:100] # Stricter cutoff for speed
        
        cands = final_pool + rescue_pool
        cands = cluster_replica_families(cands, est_scale)
        
        # Keep Top 150 combined for expensive features
        cands.sort(key=lambda x: x['corr_score'], reverse=True)
        cands = cands[:150]
        
        if len(cands) == 0:
            continue
            
        periodicity = estimate_periodicity_from_corr(corr_plane)
        pitch_x = periodicity['pitch_x']
        pitch_y = periodicity['pitch_y']
        mode_strong = 1 if periodicity['mode'] == 'STRONG' else 0
        
        # 4. Feature Extraction
        rows = []
        for c in cands:
            cx, cy = c['cx'], c['cy']
            px, py = c['peak_x'], c['peak_y']
            
            ctx = verify_candidate_context(ref, search, cx, cy, est_scale, est_theta)
            phase_pen = verify_phase_consistency(search, best_template, px, py)
            neigh_cons = compute_neighborhood_consistency(search, best_template, px, py, pitch_x, pitch_y)
            grad_ncc = compute_gradient_ncc(search, best_template, px, py)
            
            rows.append({
                'pair_id': pid,
                'cx': cx, 'cy': cy,
                'corr_score': c['corr_score'],
                'psr': c.get('psr', 0),
                'context_128': ctx['s128'],
                'context_combined': ctx['combined'],
                'phase_penalty': phase_pen,
                'family_population': c.get('family_population', 1),
                'dist_to_center': c.get('dist_to_center', 0.0),
                'neigh_cons': neigh_cons,
                'grad_ncc': grad_ncc,
                'queue': c['queue'],
                'gt_found': row['gt_found'],
                'gt_x': row['gt_x'],
                'gt_y': row['gt_y'],
                'mode_strong': mode_strong
            })
            
        df = pd.DataFrame(rows)
        feature_cols = ['corr_score', 'psr', 'context_128', 'context_combined', 'phase_penalty', 
                       'dist_to_center', 'neigh_cons', 'grad_ncc']
        for col in feature_cols:
            df[col + '_rel'] = df[col] - df[col].median()
        df['family_ratio'] = df['family_population'] / len(cands)
        
        # V25 Rank
        X_rank = df[ranker_model['features']]
        df['v25_ml_score'] = ranker_model['model'].predict_proba(X_rank)[:, 1]
        
        # Label true candidates (for present pairs)
        if row['gt_found'] == 1:
            df['is_correct'] = (np.abs(df['cx'] - row['gt_x']) <= max(25, tw*0.25)) & (np.abs(df['cy'] - row['gt_y']) <= max(25, th*0.25))
            df['is_correct'] = df['is_correct'].astype(int)
        else:
            df['is_correct'] = 0
            
        all_features.append(df)
        
    final_df = pd.concat(all_features, ignore_index=True)
    final_df.to_csv('phase2/V26_CHAMPIONSHIP/v26_extracted_features.csv', index=False)
    print("V26 feature extraction complete.")

if __name__ == '__main__':
    extract_v26_features()
