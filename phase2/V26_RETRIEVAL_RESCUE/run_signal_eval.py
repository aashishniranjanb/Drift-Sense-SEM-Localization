import os
import sys
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('phase2')
sys.path.append('fallbacks')
sys.path.append('team/akhilesh-localization')

from pose_fallback import perform_pose_fallback_search
from candidate_extractor import extract_nms_fast
from inference_phase2 import verify_candidate_context, verify_phase_consistency
from phase2.V25_CHAMPIONSHIP.feature_extractors import compute_gradient_ncc

pairs = pd.read_csv('data/phase2_dev/pairs.csv')
tax = pd.read_csv('phase2/V26_RETRIEVAL_RESCUE/retrieval_failure_taxonomy.csv')
out200 = pd.read_csv('phase2/V26_RETRIEVAL_RESCUE/outside200.csv')

ranks_ncc = []
ranks_psr = []
ranks_center = []
ranks_context = []
ranks_gradient = []
ranks_phase = []

for idx, row in tqdm(out200.iterrows(), total=len(out200)):
    pid = row['pair_id']
    p_row = pairs[pairs['pair_id'] == pid].iloc[0]
    gt_x = float(p_row['gt_x'])
    gt_y = float(p_row['gt_y'])
    
    ref = cv2.imread(os.path.join('data/phase2_dev', p_row['reference_path']), 0)
    search = cv2.imread(os.path.join('data/phase2_dev', p_row['search_path']), 0)
    
    pose = perform_pose_fallback_search(ref, search)
    corr_plane = pose['corr_plane']
    best_template = pose['best_template']
    est_scale = pose['best_scale']
    est_theta = pose['best_theta']
    tw, th = best_template.shape[::-1]
    
    sh, sw = search.shape[:2]
    cx_s, cy_s = sw / 2.0, sh / 2.0
    
    # Extract raw candidates with radius=2 to allow close peaks (resolves NMS suppression)
    raw_cands = extract_nms_fast(corr_plane, tw, th, max_k=1000, r=2)
    
    # Add GT exact spot if not present
    gt_px = int(round(gt_x - tw / 2.0))
    gt_py = int(round(gt_y - th / 2.0))
    gt_cand = {
        'peak_x': gt_px,
        'peak_y': gt_py,
        'cx': gt_x,
        'cy': gt_y,
        'corr_score': float(corr_plane[max(0, min(corr_plane.shape[0]-1, gt_py)), max(0, min(corr_plane.shape[1]-1, gt_px))]),
        'is_gt': True
    }
    
    pool = [gt_cand]
    for c in raw_cands:
        c_copy = dict(c)
        c_copy['is_gt'] = bool(np.hypot(c['cx'] - gt_x, c['cy'] - gt_y) <= 5.0)
        pool.append(c_copy)
        
    # Rank by NCC
    pool.sort(key=lambda x: x['corr_score'], reverse=True)
    rk_ncc = next((i+1 for i, c in enumerate(pool) if c['is_gt']), 1001)
    ranks_ncc.append(rk_ncc)
    
    # Rank by center distance
    pool.sort(key=lambda x: np.hypot(x['cx'] - cx_s, x['cy'] - cy_s))
    rk_center = next((i+1 for i, c in enumerate(pool) if c['is_gt']), 1001)
    ranks_center.append(rk_center)
    
    # For expensive signals (context, gradient, phase), rank among top 100 NCC pool
    top100 = pool[:100]
    if not any(c['is_gt'] for c in top100):
        top100.append(gt_cand)
        
    for c in top100:
        ctx = verify_candidate_context(ref, search, c['cx'], c['cy'], est_scale, est_theta)
        grad = compute_gradient_ncc(search, best_template, c['peak_x'], c['peak_y'])
        phase = verify_phase_consistency(search, best_template, c['peak_x'], c['peak_y'])
        c['context'] = ctx['combined']
        c['gradient'] = grad
        c['phase'] = -phase
        
    top100.sort(key=lambda x: x['context'], reverse=True)
    rk_ctx = next((i+1 for i, c in enumerate(top100) if c['is_gt']), 1001)
    ranks_context.append(rk_ctx)
    
    top100.sort(key=lambda x: x['gradient'], reverse=True)
    rk_grad = next((i+1 for i, c in enumerate(top100) if c['is_gt']), 1001)
    ranks_gradient.append(rk_grad)
    
    top100.sort(key=lambda x: x['phase'], reverse=True)
    rk_phase = next((i+1 for i, c in enumerate(top100) if c['is_gt']), 1001)
    ranks_phase.append(rk_phase)

signals_df = pd.DataFrame([
    {
        'signal': 'NCC',
        'top10': sum(1 for r in ranks_ncc if r <= 10),
        'top20': sum(1 for r in ranks_ncc if r <= 20),
        'top50': sum(1 for r in ranks_ncc if r <= 50),
        'top100': sum(1 for r in ranks_ncc if r <= 100),
        'top200': sum(1 for r in ranks_ncc if r <= 200)
    },
    {
        'signal': 'Gradient_NCC',
        'top10': sum(1 for r in ranks_gradient if r <= 10),
        'top20': sum(1 for r in ranks_gradient if r <= 20),
        'top50': sum(1 for r in ranks_gradient if r <= 50),
        'top100': sum(1 for r in ranks_gradient if r <= 100),
        'top200': sum(1 for r in ranks_gradient if r <= 200)
    },
    {
        'signal': 'Context_Combined',
        'top10': sum(1 for r in ranks_context if r <= 10),
        'top20': sum(1 for r in ranks_context if r <= 20),
        'top50': sum(1 for r in ranks_context if r <= 50),
        'top100': sum(1 for r in ranks_context if r <= 100),
        'top200': sum(1 for r in ranks_context if r <= 200)
    },
    {
        'signal': 'Phase_Consistency',
        'top10': sum(1 for r in ranks_phase if r <= 10),
        'top20': sum(1 for r in ranks_phase if r <= 20),
        'top50': sum(1 for r in ranks_phase if r <= 50),
        'top100': sum(1 for r in ranks_phase if r <= 100),
        'top200': sum(1 for r in ranks_phase if r <= 200)
    },
    {
        'signal': 'Center_Proximity',
        'top10': sum(1 for r in ranks_center if r <= 10),
        'top20': sum(1 for r in ranks_center if r <= 20),
        'top50': sum(1 for r in ranks_center if r <= 50),
        'top100': sum(1 for r in ranks_center if r <= 100),
        'top200': sum(1 for r in ranks_center if r <= 200)
    }
])

signals_df.to_csv('phase2/V26_RETRIEVAL_RESCUE/retrieval_signal_coverage.csv', index=False)
print('\nSignal Coverage on 35 Outside-200 Cases:')
print(signals_df.to_string(index=False))
