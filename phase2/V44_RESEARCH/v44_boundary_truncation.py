import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('phase2/V26_CHAMPIONSHIP/v26_extracted_features.csv')
df = df[df['gt_found'] == 1]
df = df[df['queue'] == 'V25']
df = df.sort_values(by=['pair_id', 'v25_ml_score'], ascending=[True, False])

# Group A: Top-1 wrong, GT in Top-20
group_A = []
for pair_id, group in df.groupby('pair_id', sort=False):
    cands = group.to_dict('records')
    if len(cands) == 0: continue
    if cands[0]['is_correct'] == 0:
        if any(c['is_correct'] == 1 for c in cands[1:20]):
            group_A.append(pair_id)

# Safety Set (Group B): Top-1 correct
group_B = []
for pair_id, group in df.groupby('pair_id', sort=False):
    cands = group.to_dict('records')
    if len(cands) > 0 and cands[0]['is_correct'] == 1:
        group_B.append(pair_id)

print(f"Group A (Hard): {len(group_A)}")
print(f"Group B (Safety): {len(group_B)}")

def evaluate_pair(pair_id, cands):
    # Full candidate pool for the image
    cand_pts = np.array([[c['cx'], c['cy']] for c in cands])
    
    # 1. Discover constellation vectors
    dx = cand_pts[:, 0, None] - cand_pts[None, :, 0]
    dy = cand_pts[:, 1, None] - cand_pts[None, :, 1]
    dx = dx.flatten()
    dy = dy.flatten()
    
    # filter out 0,0
    mask = (dx != 0) | (dy != 0)
    dx = dx[mask]
    dy = dy[mask]
    
    # 2D Histogram to find robust displacement vectors
    # bin size 4x4
    bins = np.arange(-1000, 1000, 4)
    H, xedges, yedges = np.histogram2d(dx, dy, bins=bins)
    
    # Find robust vectors (frequency >= 5)
    # Actually, if the image has a 10x10 grid, the (1,0) vector appears 90 times!
    # But a random noise vector appears 1 time.
    y_idx, x_idx = np.where(H >= min(5, len(cands) * 0.05)) # scale threshold with pool size
    
    V = []
    for yi, xi in zip(y_idx, x_idx):
        vx = xedges[xi] + 2 # center of bin
        vy = yedges[yi] + 2
        V.append(np.array([vx, vy]))
        
    V = np.array(V)
    
    # 2. Evaluate candidates
    res_cands = []
    for c in cands:
        cx, cy = c['cx'], c['cy']
        f = {}
        
        # V44-A features (for composite)
        f['normalized_x'] = cx / 1000.0
        f['normalized_y'] = cy / 1000.0
        f['center_distance'] = np.hypot(cx - 500, cy - 500)
        f['min_boundary_distance'] = min(cx, 1000-cx, cy, 1000-cy)
        
        # Constellation asymmetry
        dists = np.hypot(cand_pts[:, 0] - cx, cand_pts[:, 1] - cy)
        mask_self = dists > 0.1
        other_pts = cand_pts[mask_self]
        dists2 = dists[mask_self]
        idx8 = np.argsort(dists2)[:8]
        if len(idx8) >= 8:
            centroid_x = np.mean(other_pts[idx8, 0])
            centroid_y = np.mean(other_pts[idx8, 1])
            f['constellation_asymmetry'] = np.hypot(centroid_x - cx, centroid_y - cy)
        else:
            f['constellation_asymmetry'] = 0
            
        # V44-B features
        expected = 0
        observed = 0
        
        if len(V) > 0:
            target_pts = V + np.array([cx, cy])
            valid = (target_pts[:, 0] >= 0) & (target_pts[:, 0] <= 1000) & (target_pts[:, 1] >= 0) & (target_pts[:, 1] <= 1000)
            
            valid_targets = target_pts[valid]
            expected = len(valid_targets)
            
            for tp in valid_targets:
                if np.min(np.hypot(cand_pts[:, 0] - tp[0], cand_pts[:, 1] - tp[1])) < 8:
                    observed += 1
                    
        f['expected_replica_count'] = expected
        f['observed_replica_count'] = observed
        f['missing_replica_count'] = expected - observed
        f['boundary_consistency'] = observed / max(1, expected)
        f['replica_visibility_ratio'] = f['boundary_consistency']
        f['boundary_truncation_score'] = expected
        
        c_res = dict(c)
        c_res.update(f)
        res_cands.append(c_res)
        
    return res_cands

# Process Group A
group_A_results = []
for pair_id in group_A:
    cands = df[df['pair_id'] == pair_id].to_dict('records')
    res_cands = evaluate_pair(pair_id, cands)
    top1 = res_cands[0]
    gt_cand = next(c for c in res_cands[1:20] if c['is_correct'] == 1)
    
    group_A_results.append({
        'pair_id': pair_id,
        'top1': top1,
        'gt': gt_cand
    })

# Evaluate V44-B metrics on Group A
print("\n--- V44-B RESULT ---")
print(f"Group A: N = {len(group_A)}")
print("\nFeature                     GT win")
print("-" * 34)

metrics = ['boundary_consistency', 'replica_visibility_ratio', 'observed_replica_count', 'missing_replica_count', 'boundary_truncation_score']
for m in metrics:
    gt_vals = np.array([r['gt'][m] for r in group_A_results])
    top1_vals = np.array([r['top1'][m] for r in group_A_results])
    
    win_g = np.sum(gt_vals > top1_vals) / len(group_A)
    win_l = np.sum(gt_vals < top1_vals) / len(group_A)
    best = max(win_g, win_l)
    dir_str = "GREATER" if win_g > win_l else "LESS"
    
    print(f"{m:<25} {best*100:.1f}% ({dir_str})")

# Composite Score
def compute_composite(c):
    # Normalize features using arbitrary safe ranges for discovery
    # boundary_consistency: [0, 1]
    # constellation_asymmetry: ~ [0, 50]
    # center_distance: ~ [0, 700]
    
    n_bc = c['boundary_consistency']
    n_ca = min(c['constellation_asymmetry'] / 25.0, 1.0)
    n_cd = min(c['center_distance'] / 500.0, 1.0)
    
    return n_bc * 2.0 + n_ca * 1.0 - n_cd * 0.5 + c['v25_ml_score'] * 1.0

rescues = 0
for r in group_A_results:
    if compute_composite(r['gt']) > compute_composite(r['top1']):
        rescues += 1
print(f"\nCOMPOSITE:\nGT win = {rescues/len(group_A)*100:.1f}%")
print(f"RESCUES:\nV25 Top-1 GT = 0\nV44 Top-1 GT = {rescues}")

# Process Group B (Safety)
demotions = 0
for pair_id in group_B:
    cands = df[df['pair_id'] == pair_id].to_dict('records')
    res_cands = evaluate_pair(pair_id, cands)
    
    top1 = res_cands[0]
    best_c = top1
    best_score = compute_composite(top1)
    
    for c in res_cands[1:20]:
        s = compute_composite(c)
        if s > best_score + 0.15: # safety margin!
            best_score = s
            best_c = c
            
    if best_c != top1 and best_c['is_correct'] == 0:
        demotions += 1

print(f"\nSAFETY SET:")
print(f"correct V25 candidates = {len(group_B)}")
print(f"V44 would demote = {demotions}")
print(f"demotion rate = {demotions/len(group_B)*100:.1f}%")

if rescues/len(group_A) > 0.7 and demotions/len(group_B) < 0.05:
    print("\nVERDICT:\nKEEP")
else:
    print("\nVERDICT:\nKILL (or tune)")

