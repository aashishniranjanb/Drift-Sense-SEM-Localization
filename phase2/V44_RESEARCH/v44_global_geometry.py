import pandas as pd
import numpy as np

# Read candidate features
df = pd.read_csv('phase2/V26_CHAMPIONSHIP/v26_extracted_features.csv')
df = df[df['gt_found'] == 1]
df = df[df['queue'] == 'V25']
df = df.sort_values(by=['pair_id', 'v25_ml_score'], ascending=[True, False])

# Collect eligible pairs: Top-1 is WRONG, GT is in Top-20
group_A_pairs = []
for pair_id, group in df.groupby('pair_id', sort=False):
    cands = group.to_dict('records')
    if len(cands) == 0: continue
    top1 = cands[0]
    if top1['is_correct'] == 0:
        gt_cand = next((c for c in cands[1:20] if c['is_correct'] == 1), None)
        if gt_cand:
            group_A_pairs.append(pair_id)

print(f"Eligible Group A pairs: {len(group_A_pairs)}")

# We need the canvas dimensions. By default search images are 1000x1000.
w, h = 1000, 1000

results = []
for pair_id in group_A_pairs:
    cands = df[df['pair_id'] == pair_id].to_dict('records')
    
    # We will compute neighbor stats using the full candidate pool for this pair
    cand_pts = np.array([[c['cx'], c['cy']] for c in cands])
    
    top1 = cands[0]
    gt_cand = next(c for c in cands[1:20] if c['is_correct'] == 1)
    
    def get_features(c):
        cx, cy = c['cx'], c['cy']
        f = {}
        
        f['normalized_x'] = cx / w
        f['normalized_y'] = cy / h
        
        f['center_distance'] = np.hypot(cx - w/2, cy - h/2)
        
        f['boundary_left'] = cx
        f['boundary_right'] = w - cx
        f['boundary_top'] = cy
        f['boundary_bottom'] = h - cy
        
        f['min_boundary_distance'] = min(f['boundary_left'], f['boundary_right'], f['boundary_top'], f['boundary_bottom'])
        
        # Neighbor distances
        dists = np.hypot(cand_pts[:, 0] - cx, cand_pts[:, 1] - cy)
        # Exclude self
        dists = dists[dists > 0.1]
        
        if len(dists) > 0:
            f['neighbor_count_10'] = np.sum(dists < 10)
            f['neighbor_count_20'] = np.sum(dists < 20)
            f['neighbor_count_50'] = np.sum(dists < 50)
            
            dists_sorted = np.sort(dists)
            f['nearest_neighbor_distance'] = dists_sorted[0] if len(dists_sorted) > 0 else 0
            f['2nd_neighbor_distance'] = dists_sorted[1] if len(dists_sorted) > 1 else 0
            f['3rd_neighbor_distance'] = dists_sorted[2] if len(dists_sorted) > 2 else 0
            f['neighbor_distance_std'] = np.std(dists_sorted[:5]) if len(dists_sorted) >= 5 else 0
            
            # Constellation asymmetry: centroid of nearest 8 neighbors vs candidate center
            n8 = np.argsort(dists)[:8]
            n8_pts = cand_pts[n8] # wait, cand_pts is full array, we didn't filter self in index!
            
            # Better to use a clean mask
            mask = np.ones(len(cand_pts), dtype=bool)
            mask[np.argmin(np.hypot(cand_pts[:, 0] - cx, cand_pts[:, 1] - cy))] = False
            other_pts = cand_pts[mask]
            
            dists2 = np.hypot(other_pts[:, 0] - cx, other_pts[:, 1] - cy)
            idx8 = np.argsort(dists2)[:8]
            if len(idx8) >= 8:
                centroid_x = np.mean(other_pts[idx8, 0])
                centroid_y = np.mean(other_pts[idx8, 1])
                f['constellation_asymmetry'] = np.hypot(centroid_x - cx, centroid_y - cy)
            else:
                f['constellation_asymmetry'] = 0
        else:
            f['neighbor_count_10'] = 0
            f['neighbor_count_20'] = 0
            f['neighbor_count_50'] = 0
            f['nearest_neighbor_distance'] = 0
            f['2nd_neighbor_distance'] = 0
            f['3rd_neighbor_distance'] = 0
            f['neighbor_distance_std'] = 0
            f['constellation_asymmetry'] = 0
            
        return f

    f_top1 = get_features(top1)
    f_gt = get_features(gt_cand)
    
    res = {'pair_id': pair_id}
    for k in f_top1.keys():
        res[f'top1_{k}'] = f_top1[k]
        res[f'gt_{k}'] = f_gt[k]
        
    results.append(res)

out_df = pd.DataFrame(results)
out_df.to_csv('phase2/V44_RESEARCH/v44_results.csv', index=False)

# Evaluation
features = [k[3:] for k in out_df.columns if k.startswith('gt_')]
print("\n--- V44-A PAIRWISE EVALUATION ---")
print(f"Total evaluated: {len(out_df)}")

report = []

for f in features:
    gt_vals = out_df[f'gt_{f}']
    top1_vals = out_df[f'top1_{f}']
    
    # We define a "win" depending on the feature's natural direction.
    # But since we don't know the direction yet, we will measure:
    # "GT is strictly GREATER than Top-1"
    # "GT is strictly LESS than Top-1"
    
    gt_greater = (gt_vals > top1_vals).sum()
    gt_less = (gt_vals < top1_vals).sum()
    ties = (gt_vals == top1_vals).sum()
    
    gt_greater_pct = gt_greater / len(out_df)
    gt_less_pct = gt_less / len(out_df)
    ties_pct = ties / len(out_df)
    
    best_win = max(gt_greater_pct, gt_less_pct)
    direction = "GREATER" if gt_greater_pct > gt_less_pct else "LESS"
    
    if best_win > 0:
        report.append({
            'feature': f,
            'max_win_rate': best_win,
            'direction_for_gt_win': direction,
            'ties': ties_pct
        })

report_df = pd.DataFrame(report).sort_values(by='max_win_rate', ascending=False)
for _, r in report_df.iterrows():
    print(f"{r['feature']:<25} | GT wins: {r['max_win_rate']*100:.1f}% ({r['direction_for_gt_win']}) | Ties: {r['ties']*100:.1f}%")

