"""
V24-C Response Surface Exploration
Pick 3 failure cases where GT was in the pool but V18-C picked a replica.
Perturb scale and rotation for both candidates to see if the response surface differs.
"""
import os
import cv2
import pandas as pd
import numpy as np

def compute_ncc(ref_img, search_img, cx, cy, scale, theta):
    """Compute local NCC at a specific scale/rotation and center"""
    # Rotate reference
    h, w = ref_img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), theta, 1.0)
    ref_rot = cv2.warpAffine(ref_img, M, (w, h), flags=cv2.INTER_LINEAR)
    
    # Scale reference to match search image
    sw = int(round(w / scale))
    sh = int(round(h / scale))
    if sw <= 1 or sh <= 1: return 0.0
    ref_scaled = cv2.resize(ref_rot, (sw, sh), interpolation=cv2.INTER_AREA)
    
    # Normalize reference
    ref_norm = (ref_scaled - np.mean(ref_scaled)) / (np.std(ref_scaled) + 1e-6)
    
    # Extract search crop
    y1, y2 = max(0, int(cy - sh/2)), min(search_img.shape[0], int(cy + sh/2))
    x1, x2 = max(0, int(cx - sw/2)), min(search_img.shape[1], int(cx + sw/2))
    
    if (y2-y1) < sh or (x2-x1) < sw:
        # pad if near edge (simplified handling)
        return 0.0
        
    search_crop = search_img[y1:y1+sh, x1:x1+sw]
    search_norm = (search_crop - np.mean(search_crop)) / (np.std(search_crop) + 1e-6)
    
    ncc = np.mean(ref_norm * search_norm)
    return ncc

def explore_response_surface():
    df = pd.read_csv("phase2/V22_CHAMPIONSHIP/results/candidate_pool_features.csv")
    gt = pd.read_csv("data/phase2_dev/pairs.csv")
    
    # Find a pair where GT is correct but V18-C failed
    # We'll just pick pair_008 or similar that is periodic.
    # Let's filter to PRESENT pairs
    gt_present = gt[gt.gt_found == 1]
    
    test_pairs = ["pair_014", "pair_022", "pair_031"] # Known periodic failures typically
    
    for pair_id in test_pairs:
        if pair_id not in df.pair_id.values:
            continue
            
        print(f"\\n=== Exploring {pair_id} ===")
        pool = df[df.pair_id == pair_id]
        
        gt_cand = pool[pool.is_correct == 1]
        if len(gt_cand) == 0:
            print("GT not in pool.")
            continue
            
        gt_cand = gt_cand.iloc[0]
        
        # False winning candidate (highest corr_score that is not GT)
        false_cand = pool[pool.is_correct == 0].sort_values("corr_score", ascending=False).iloc[0]
        
        print(f"GT Cand: corr={gt_cand.corr_score:.4f}, cx={gt_cand.cx:.1f}, cy={gt_cand.cy:.1f}")
        print(f"False Cand: corr={false_cand.corr_score:.4f}, cx={false_cand.cx:.1f}, cy={false_cand.cy:.1f}")
        
        # Load images
        pair_row = gt[gt.pair_id == pair_id].iloc[0]
        ref_path = os.path.join("data/phase2_dev", pair_row.reference_path)
        search_path = os.path.join("data/phase2_dev", pair_row.search_path)
        ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        search = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
        est_scale = pair_row.gt_scale
        est_theta = pair_row.gt_theta
        
        # Perturbations
        scales = [est_scale * f for f in [0.95, 0.98, 1.0, 1.02, 1.05]]
        thetas = [est_theta + d for d in [-2.0, -0.5, 0.0, 0.5, 2.0]]
        
        print("  Scale perturbations (theta=gt):")
        gt_s_vals = []
        false_s_vals = []
        for s in scales:
            gt_ncc = compute_ncc(ref, search, gt_cand.cx, gt_cand.cy, s, est_theta)
            false_ncc = compute_ncc(ref, search, false_cand.cx, false_cand.cy, s, est_theta)
            gt_s_vals.append(f"{gt_ncc:.3f}")
            false_s_vals.append(f"{false_ncc:.3f}")
        print("    Scales:", [round(s, 2) for s in scales])
        print("    GT NCC:  ", gt_s_vals)
        print("    False NCC:", false_s_vals)
        
        print("  Theta perturbations (scale=gt):")
        gt_t_vals = []
        false_t_vals = []
        for t in thetas:
            gt_ncc = compute_ncc(ref, search, gt_cand.cx, gt_cand.cy, est_scale, t)
            false_ncc = compute_ncc(ref, search, false_cand.cx, false_cand.cy, est_scale, t)
            gt_t_vals.append(f"{gt_ncc:.3f}")
            false_t_vals.append(f"{false_ncc:.3f}")
        print("    Thetas:", [round(t, 2) for t in thetas])
        print("    GT NCC:  ", gt_t_vals)
        print("    False NCC:", false_t_vals)

if __name__ == "__main__":
    explore_response_surface()
