import os
import sys
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.append('phase2')
sys.path.append('fallbacks')
sys.path.append('team/akhilesh-localization')
sys.path.append('production_engine')

from pose_fallback import perform_pose_fallback_search
from candidate_extractor import extract_candidates_akhilesh
from inference_phase2 import cluster_replica_families, verify_candidate_context, verify_phase_consistency
from phase2.V25_CHAMPIONSHIP.periodicity import estimate_periodicity_from_corr
from phase2.V25_CHAMPIONSHIP.feature_extractors import compute_neighborhood_consistency, compute_gradient_ncc

def build_dataset():
    df = pd.read_csv('data/phase2_dev/pairs.csv')
    
    rows = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        pair_id = row['pair_id']
        gt_x = float(row['gt_x'])
        gt_y = float(row['gt_y'])
        gt_found = int(row['gt_found'])
        
        ref_path = os.path.join('data/phase2_dev', row['reference_path'])
        search_path = os.path.join('data/phase2_dev', row['search_path'])
        
        ref = cv2.imread(ref_path, 0)
        search = cv2.imread(search_path, 0)
        
        if ref is None or search is None:
            continue
            
        pose = perform_pose_fallback_search(ref, search)
        corr_plane = pose['corr_plane']
        best_template = pose['best_template']
        est_scale = pose['best_scale']
        est_theta = pose['best_theta']
        tw, th = best_template.shape[::-1]
        
        periodicity = estimate_periodicity_from_corr(corr_plane)
        pitch_x = periodicity['pitch_x']
        pitch_y = periodicity['pitch_y']
        
        cands = extract_candidates_akhilesh(corr_plane, tw, th, ref, search, est_scale, est_theta, max_final_k=200)
        cands = cluster_replica_families(cands, est_scale)
        
        for rank_idx, c in enumerate(cands):
            cx = c['cx']
            cy = c['cy']
            px = c['peak_x']
            py = c['peak_y']
            
            # Ground truth label
            label = 0
            if gt_found == 1:
                dist = np.sqrt((cx - gt_x)**2 + (cy - gt_y)**2)
                if dist <= 5.0:
                    label = 1
                    
            ctx = verify_candidate_context(ref, search, cx, cy, est_scale, est_theta)
            phase_pen = verify_phase_consistency(search, best_template, px, py)
            
            neigh_cons = compute_neighborhood_consistency(search, best_template, px, py, pitch_x, pitch_y)
            grad_ncc = compute_gradient_ncc(search, best_template, px, py)
            
            rows.append({
                'pair_id': pair_id,
                'gt_found': gt_found,
                'label': label,
                'rank_idx': rank_idx,
                'corr_score': c['corr_score'],
                'psr': c.get('psr', 0),
                'context_128': ctx['s128'],
                'context_combined': ctx['combined'],
                'phase_penalty': phase_pen,
                'family_population': c.get('family_population', 1),
                'dist_to_center': c.get('dist_to_center', 0.0),
                'neigh_cons': neigh_cons,
                'grad_ncc': grad_ncc,
                'pitch_x': pitch_x,
                'pitch_y': pitch_y,
                'periodicity_mode': periodicity['mode']
            })
            
    out_df = pd.DataFrame(rows)
    out_df.to_csv('phase2/V25_CHAMPIONSHIP/v25_train_features.csv', index=False)
    print("Saved to v25_train_features.csv")

if __name__ == '__main__':
    build_dataset()
