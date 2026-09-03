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
group_B_pairs = []

for pair_id, group in df.groupby('pair_id', sort=False):
    cands = group.to_dict('records')
    if len(cands) == 0: continue
    top1 = cands[0]
    if top1['is_correct'] == 0:
        if any(c['is_correct'] == 1 for c in cands[1:20]):
            group_A_pairs.append(pair_id)
    elif top1['is_correct'] == 1:
        group_B_pairs.append(pair_id)

eval_pairs = group_A_pairs[:20] + group_B_pairs[:10]

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
        
    h, w = search.shape
    
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
        
        # Curvature: simple discrete 2nd derivative at 1.00
        idx_100 = scales.index(1.00)
        c_m2 = c_scores[idx_100 - 2]
        c_m1 = c_scores[idx_100 - 1]
        c_0  = c_scores[idx_100]
        c_p1 = c_scores[idx_100 + 1]
        c_p2 = c_scores[idx_100 + 2]
        
        curvature = (c_p1 + c_m1 - 2*c_0) # should be negative if it's a peak
        
        results.append({
            'pair_id': pair_id,
            'is_correct': c['is_correct'],
            'v25_ml_score': c['v25_ml_score'],
            'peak_strength': max(c_scores),
            'curvature': curvature,
            'scale_std': np.std(c_scores),
            'best_scale': scales[np.argmax(c_scores)]
        })

out_df = pd.DataFrame(results)

real_df = out_df[out_df['is_correct'] == 1]
wrong_df = out_df[(out_df['is_correct'] == 0) & (out_df['v25_ml_score'] > 0.5)]

print("\n--- V41 RESULT ---")
print(f"REAL MATCHES: N = {len(real_df)}")
print(f"PERIODIC WRONG: N = {len(wrong_df)}")

print("\nCurvature (more negative is sharper peak):")
print(f"REAL = {real_df['curvature'].mean():.4f}")
print(f"WRONG = {wrong_df['curvature'].mean():.4f}")

print("\nScale response std (higher means it cares about scale):")
print(f"REAL = {real_df['scale_std'].mean():.4f}")
print(f"WRONG = {wrong_df['scale_std'].mean():.4f}")
