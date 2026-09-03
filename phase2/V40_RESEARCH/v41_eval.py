import pandas as pd
import numpy as np
import cv2
import os
import sys

sys.path.append('phase2')
sys.path.append('fallbacks')
sys.path.append('team/akhilesh-localization')

from pose_fallback import perform_pose_fallback_search

df = pd.read_csv('phase2/V26_CHAMPIONSHIP/v26_extracted_features.csv')
df = df[df['gt_found'] == 1]
df = df[df['queue'] == 'V25']
df = df.sort_values(by=['pair_id', 'v25_ml_score'], ascending=[True, False])

pairs_df = pd.read_csv('data/phase2_dev/pairs.csv')

group_A_pairs = []
for pair_id, group in df.groupby('pair_id', sort=False):
    cands = group.to_dict('records')
    if len(cands) == 0: continue
    top1 = cands[0]
    if top1['is_correct'] == 0 and any(c['is_correct'] == 1 for c in cands[1:20]):
        group_A_pairs.append(pair_id)

eval_pairs = group_A_pairs[:20]
results = []

for pair_id in eval_pairs:
    cands = df[df['pair_id'] == pair_id].to_dict('records')[:10]
    
    pinfo = pairs_df[pairs_df['pair_id'] == pair_id].iloc[0]
    ref = cv2.imread(os.path.join('data/phase2_dev', pinfo['reference_path']), 0)
    search = cv2.imread(os.path.join('data/phase2_dev', pinfo['search_path']), 0)
    
    pose = perform_pose_fallback_search(ref, search)
    temp = pose['best_template']
    
    scales = [0.96, 0.98, 1.00, 1.02, 1.04]
    scores_by_scale = {}
    for s in scales:
        tw = int(temp.shape[1] * s)
        th = int(temp.shape[0] * s)
        scaled_temp = cv2.resize(temp, (tw, th), interpolation=cv2.INTER_AREA)
        corr = cv2.matchTemplate(search.astype(np.float32), scaled_temp.astype(np.float32), cv2.TM_CCOEFF_NORMED)
        scores_by_scale[s] = (corr, tw, th)
        
    for c in cands:
        c_scores = []
        for s in scales:
            corr, tw, th = scores_by_scale[s]
            px, py = int(c['cx'] - tw/2.0), int(c['cy'] - th/2.0)
            if px < 0 or py < 0 or px >= corr.shape[1] or py >= corr.shape[0]:
                c_scores.append(0.0)
            else:
                c_scores.append(corr[py, px])
                
        if max(c_scores) == 0.0: continue
        
        idx_100 = scales.index(1.00)
        curvature = (c_scores[idx_100 - 1] + c_scores[idx_100 + 1] - 2*c_scores[idx_100])
        
        results.append({
            'pair_id': pair_id,
            'is_correct': c['is_correct'],
            'v25_ml_score': c['v25_ml_score'],
            'curvature': curvature,
            'scale_std': np.std(c_scores)
        })

out_df = pd.DataFrame(results)
out_df.to_csv('phase2/V40_RESEARCH/v41_data.csv', index=False)

rescued = 0
for pair_id, group in out_df.groupby('pair_id'):
    cands = group.to_dict('records')
    top1 = cands[0]
    gt_cand = next((c for c in cands if c['is_correct'] == 1), None)
    if gt_cand:
        # Check if GT has more negative curvature than top1
        if gt_cand['curvature'] < top1['curvature'] - 0.05:
            rescued += 1
            
print(f"GT has significantly sharper scale peak in {rescued}/{len(group_A_pairs[:20])} cases")
