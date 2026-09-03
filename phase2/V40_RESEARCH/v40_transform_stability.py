import pandas as pd
import numpy as np
import cv2
import os
import sys

sys.path.append('phase2')
sys.path.append('fallbacks')
sys.path.append('team/akhilesh-localization')

from pose_fallback import perform_pose_fallback_search

def get_transforms(img):
    img = img.astype(np.float32)
    
    orig = img
    norm = (img - np.mean(img)) / (np.std(img) + 1e-6)
    blur = cv2.GaussianBlur(img, (5, 5), 1.5)
    hp = img - blur
    
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)
    
    sharp = img + 1.0 * hp
    
    return {
        'orig': orig,
        'norm': norm,
        'hp': hp,
        'grad': grad,
        'blur': blur,
        'sharp': sharp
    }

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
        gt_cand = next((c for c in cands[1:20] if c['is_correct'] == 1), None)
        if gt_cand:
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
    
    t_search = get_transforms(search)
    t_temp = get_transforms(temp)
    
    th, tw = temp.shape
    h, w = search.shape
    
    scores = {}
    for k in t_search.keys():
        s_img = t_search[k].astype(np.float32)
        t_img = t_temp[k].astype(np.float32)
        corr = cv2.matchTemplate(s_img, t_img, cv2.TM_CCOEFF_NORMED)
        scores[k] = corr
        
    for c in cands:
        px, py = int(c['cx'] - tw/2.0), int(c['cy'] - th/2.0)
        
        if px < 0 or py < 0 or px >= scores['orig'].shape[1] or py >= scores['orig'].shape[0]:
            continue
            
        c_scores = {}
        for k in scores.keys():
            c_scores[k] = scores[k][py, px]
            
        res = {
            'pair_id': pair_id,
            'is_correct': c['is_correct'],
            'orig': c_scores['orig'],
            'norm': c_scores['norm'],
            'hp': c_scores['hp'],
            'grad': c_scores['grad'],
            'blur': c_scores['blur'],
            'sharp': c_scores['sharp'],
            'v25_ml_score': c['v25_ml_score']
        }
        
        vals = list(c_scores.values())
        res['mean'] = np.mean(vals)
        res['std'] = np.std(vals)
        res['range'] = np.max(vals) - np.min(vals)
        
        results.append(res)

out_df = pd.DataFrame(results)

real_df = out_df[out_df['is_correct'] == 1]

# For WRONG, we want only the periodic false matches that scored HIGH in V25.
# The Top-1 false matches in Group A:
wrong_df = out_df[(out_df['is_correct'] == 0) & (out_df['v25_ml_score'] > 0.5)]

print("\n--- V40 RESULT ---")
print(f"REAL MATCHES: N = {len(real_df)}")
print(f"PERIODIC WRONG: N = {len(wrong_df)}")

print("\nscore std:")
print(f"REAL = {real_df['std'].mean():.4f}")
print(f"WRONG = {wrong_df['std'].mean():.4f}")

print("\nscore range:")
print(f"REAL = {real_df['range'].mean():.4f}")
print(f"WRONG = {wrong_df['range'].mean():.4f}")

print("\nDelta from orig (grad - orig):")
print(f"REAL = {(real_df['grad'] - real_df['orig']).mean():.4f}")
print(f"WRONG = {(wrong_df['grad'] - wrong_df['orig']).mean():.4f}")

