import pandas as pd
import numpy as np
import cv2
import os
import sys

df = pd.read_csv('phase2/V26_CHAMPIONSHIP/v26_extracted_features.csv')
df = df[df['gt_found'] == 1]
df = df[df['queue'] == 'V25']
df = df.sort_values(by=['pair_id', 'v25_ml_score'], ascending=[True, False])

group_A = []
for pair_id, group in df.groupby('pair_id', sort=False):
    cands = group.to_dict('records')
    if len(cands) == 0: continue
    if cands[0]['is_correct'] == 0:
        if any(c['is_correct'] == 1 for c in cands[1:20]):
            group_A.append(pair_id)

group_B = []
for pair_id, group in df.groupby('pair_id', sort=False):
    cands = group.to_dict('records')
    if len(cands) > 0 and cands[0]['is_correct'] == 1:
        group_B.append(pair_id)

pairs_df = pd.read_csv('data/phase2_dev/pairs.csv')

def get_ring_features(img, cx, cy, tw, th, scale_factor):
    rw = int(tw * scale_factor)
    rh = int(th * scale_factor)
    px = int(cx - rw/2.0)
    py = int(cy - rh/2.0)
    
    h, w = img.shape
    if px < 0 or py < 0 or px + rw > w or py + rh > h:
        return None
        
    patch = img[py:py+rh, px:px+rw].astype(np.float32)
    mean_int = np.mean(patch)
    std_int = np.std(patch)
    
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)
    
    mean_grad = np.mean(grad)
    edge_density = np.sum(grad > 50) / (rw * rh)
    return np.array([mean_int, std_int, mean_grad, edge_density])

results = []
def evaluate_pair(pair_id):
    cands = df[df['pair_id'] == pair_id].to_dict('records')[:20]
    top1 = cands[0]
    
    pinfo = pairs_df[pairs_df['pair_id'] == pair_id].iloc[0]
    ref = cv2.imread(os.path.join('data/phase2_dev', pinfo['reference_path']), 0)
    search = cv2.imread(os.path.join('data/phase2_dev', pinfo['search_path']), 0)
    
    scale = top1.get('est_scale', 10.0) # V25 feature, might be under another name?
    # wait, v26 extracted features has 'est_scale' and 'est_theta'
    if 'est_scale' not in top1:
        # fallback
        scale = 10.0
        theta = 0.0
    else:
        scale = top1['est_scale']
        theta = top1['est_theta']
        
    M = cv2.getRotationMatrix2D((500, 500), theta, scale)
    ref_warped = cv2.warpAffine(ref, M, (1000, 1000), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    # Approx template size. Reference size is 215 at 1.0 scale
    tw = int(215 / scale) if scale > 0 else 215
    th = tw
    
    scales = [1.0, 1.5, 2.0, 3.0]
    ref_features = {}
    for s in scales:
        f = get_ring_features(ref_warped, 500, 500, tw, th, s)
        if f is None: f = np.zeros(4)
        ref_features[s] = f
        
    res_cands = []
    for cand in cands:
        cx, cy = cand['cx'], cand['cy']
        cand_res = dict(cand)
        for s in scales:
            cand_f = get_ring_features(search, cx, cy, tw, th, s)
            if cand_f is None:
                dist = 99999.0
                grad_dist = 99999.0
                edge_dist = 99999.0
            else:
                norm = np.maximum(ref_features[s], 1e-5)
                diff = np.abs(cand_f - ref_features[s]) / norm
                dist = np.mean(diff)
                grad_dist = diff[2]
                edge_dist = diff[3]
            cand_res[f'fingerprint_distance_{s}x'] = dist
            cand_res[f'gradient_distance_{s}x'] = grad_dist
            cand_res[f'edge_distance_{s}x'] = edge_dist
        res_cands.append(cand_res)
    return res_cands

group_A_res = []
for pair_id in group_A:
    cands = evaluate_pair(pair_id)
    top1 = cands[0]
    gt_cand = next((c for c in cands if c['is_correct'] == 1), None)
    if gt_cand:
        group_A_res.append({'top1': top1, 'gt': gt_cand})

group_B_res = []
for pair_id in group_B[:64]: 
    group_B_res.append(evaluate_pair(pair_id))

print("\n--- V45-A RESULT ---")
print(f"Group A (Hard): {len(group_A_res)}")

features = [
    'fingerprint_distance_1.0x', 'fingerprint_distance_1.5x', 'fingerprint_distance_2.0x', 'fingerprint_distance_3.0x',
    'gradient_distance_1.0x', 'gradient_distance_2.0x', 'gradient_distance_3.0x',
    'edge_distance_1.0x', 'edge_distance_2.0x', 'edge_distance_3.0x'
]

for f in features:
    gt_vals = np.array([r['gt'][f] for r in group_A_res])
    top1_vals = np.array([r['top1'][f] for r in group_A_res])
    win = np.sum(gt_vals < top1_vals) / len(group_A_res)
    print(f"{f:<30} {win*100:.1f}% (LESS)")

rescues = 0
for r in group_A_res:
    score_gt = -r['gt']['fingerprint_distance_3.0x']
    score_top1 = -r['top1']['fingerprint_distance_3.0x']
    if score_gt > score_top1: rescues += 1

print(f"\nCOMPOSITE (3x Distance Only):")
print(f"GT win = {rescues/len(group_A_res)*100:.1f}%")

demotions = 0
for cands in group_B_res:
    top1 = cands[0]
    best_c = top1
    best_score = -top1['fingerprint_distance_3.0x']
    for c in cands[1:]:
        s = -c['fingerprint_distance_3.0x']
        if s > best_score + 0.1:
            best_score = s
            best_c = c
    if best_c != top1 and best_c['is_correct'] == 0:
        demotions += 1

print(f"\nSAFETY SET (N={len(group_B_res)}):")
print(f"demotions = {demotions}")
print(f"demotion rate = {demotions/len(group_B_res)*100:.1f}%")

