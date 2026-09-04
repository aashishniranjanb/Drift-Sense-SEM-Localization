import sys
import os
import cv2
import numpy as np
import pickle
import time

sys.path.append('phase2')
sys.path.append('fallbacks')
sys.path.append('team/akhilesh-localization')

from pose_fallback import perform_pose_fallback_search
from candidate_extractor import extract_candidates_akhilesh
from inference_phase2 import cluster_replica_families, verify_candidate_context, verify_phase_consistency
from pose_refinement import refine_pose

from phase2.V25_CHAMPIONSHIP.periodicity import estimate_periodicity_from_corr
from phase2.V25_CHAMPIONSHIP.feature_extractors import compute_neighborhood_consistency, compute_gradient_ncc

# Load models at module level to avoid reloading
import pandas as pd
ranker_path = os.path.join(os.path.dirname(__file__), 'ranker.pkl')
presence_path = os.path.join(os.path.dirname(__file__), 'presence.pkl')
with open(ranker_path, 'rb') as f:
    ranker_model = pickle.load(f)
with open(presence_path, 'rb') as f:
    presence_model = pickle.load(f)

def run_v25_localization(ref_img: np.ndarray, search_img: np.ndarray, verbose=False):
    t0 = time.time()
    
    # 1. Pose Fallback (Base Correlator)
    pose = perform_pose_fallback_search(ref_img, search_img)
    corr_plane = pose['corr_plane']
    best_template = pose['best_template']
    est_scale = pose['best_scale']
    est_theta = pose['best_theta']
    tw, th = best_template.shape[::-1]
    
    # 2. Extract Top 200 Candidates
    cands = extract_candidates_akhilesh(corr_plane, tw, th, ref_img, search_img, est_scale, est_theta, max_final_k=200)
    cands = cluster_replica_families(cands, est_scale)
    
    if len(cands) == 0:
        return {'x': 0.0, 'y': 0.0, 'theta': 0.0, 'scale': 0.0, 'found': 0, 'score': 0.0}
        
    # 3. Global Periodicity
    periodicity = estimate_periodicity_from_corr(corr_plane)
    pitch_x = periodicity['pitch_x']
    pitch_y = periodicity['pitch_y']
    mode_strong = 1 if periodicity['mode'] == 'STRONG' else 0
    
    # 4. Feature Extraction
    rows = []
    for c in cands:
        cx, cy = c['cx'], c['cy']
        px, py = c['peak_x'], c['peak_y']
        
        ctx = verify_candidate_context(ref_img, search_img, cx, cy, est_scale, est_theta)
        phase_pen = verify_phase_consistency(search_img, best_template, px, py)
        neigh_cons = compute_neighborhood_consistency(search_img, best_template, px, py, pitch_x, pitch_y)
        grad_ncc = compute_gradient_ncc(search_img, best_template, px, py)
        
        rows.append({
            'corr_score': c['corr_score'],
            'psr': c.get('psr', 0),
            'context_128': ctx['s128'],
            'context_combined': ctx['combined'],
            'phase_penalty': phase_pen,
            'family_population': c.get('family_population', 1),
            'dist_to_center': c.get('dist_to_center', 0.0),
            'neigh_cons': neigh_cons,
            'grad_ncc': grad_ncc
        })
        
    df = pd.DataFrame(rows)
    feature_cols = ['corr_score', 'psr', 'context_128', 'context_combined', 'phase_penalty', 
                   'dist_to_center', 'neigh_cons', 'grad_ncc']
    for col in feature_cols:
        df[col + '_rel'] = df[col] - df[col].median()
    df['family_ratio'] = df['family_population'] / len(cands)
    
    # 5. ML Ranking
    X_rank = df[ranker_model['features']]
    rank_scores = ranker_model['model'].predict_proba(X_rank)[:, 1]
    
    for i, c in enumerate(cands):
        c['ml_score'] = rank_scores[i]
        
    cands.sort(key=lambda x: x['ml_score'], reverse=True)
    best_cand = cands[0]
    second_cand = cands[1] if len(cands) > 1 else best_cand
    
    best_idx = cands.index(best_cand)
    
    # 6. ML Presence Rejection
    X_pres = pd.DataFrame([{
        'top1_score': best_cand['ml_score'],
        'margin': best_cand['ml_score'] - second_cand['ml_score'],
        'top1_corr': df.iloc[best_idx]['corr_score'],
        'top1_ctx': df.iloc[best_idx]['context_combined'],
        'top1_neigh': df.iloc[best_idx]['neigh_cons'],
        'top1_grad': df.iloc[best_idx]['grad_ncc'],
        'mode_strong': mode_strong
    }])
    pres_score = float(presence_model['model'].predict_proba(X_pres[presence_model['features']])[0, 1])
    found = 1 if pres_score > 0.843 else 0
    
    # 7. Subpixel Refinement
    if found == 1:
        rx, ry, _, _ = refine_pose(ref_img, search_img, est_scale, est_theta, best_cand['peak_x'], best_cand['peak_y'], corr_plane)
    else:
        rx, ry, est_theta, est_scale = 0.0, 0.0, 0.0, 0.0
        
    if verbose:
        print(f"V25 Time: {time.time()-t0:.2f}s, Found: {found}, Conf: {pres_score:.3f}")
        
    return {
        'x': rx,
        'y': ry,
        'theta': est_theta,
        'scale': est_scale,
        'found': found,
        'score': pres_score
    }
