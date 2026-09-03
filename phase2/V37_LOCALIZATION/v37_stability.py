import os
import sys
import cv2
import numpy as np
import pandas as pd
import time

sys.path.append('.')
sys.path.append('phase2')
sys.path.append('fallbacks')
sys.path.append('team/akhilesh-localization')

from pose_fallback import perform_pose_fallback_search
from candidate_extractor import extract_candidates_akhilesh
from family_clustering import cluster_replica_families

def get_representations(ref_img: np.ndarray, search_img: np.ndarray):
    """
    Generate 6 representations for ref and search images:
    1. Original
    2. Intensity-normalized
    3. High-pass
    4. Gradient (Scharr)
    5. Mild Gaussian blur
    6. Mild sharpening
    """
    reps_ref = []
    reps_srch = []
    
    # 1. Original
    ref_orig = ref_img.astype(np.float32)
    srch_orig = search_img.astype(np.float32)
    reps_ref.append(ref_orig)
    reps_srch.append(srch_orig)
    
    # 2. Intensity-normalized (Min-Max)
    ref_norm = cv2.normalize(ref_orig, None, 0, 255, cv2.NORM_MINMAX)
    srch_norm = cv2.normalize(srch_orig, None, 0, 255, cv2.NORM_MINMAX)
    reps_ref.append(ref_norm)
    reps_srch.append(srch_norm)
    
    # 3. High-pass (Original - GaussianBlur)
    ref_hp = ref_orig - cv2.GaussianBlur(ref_orig, (15, 15), 3.0)
    srch_hp = srch_orig - cv2.GaussianBlur(srch_orig, (15, 15), 3.0)
    reps_ref.append(ref_hp)
    reps_srch.append(srch_hp)
    
    # 4. Gradient (Scharr Magnitude)
    gx = cv2.Scharr(ref_orig, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(ref_orig, cv2.CV_32F, 0, 1)
    ref_grad = cv2.magnitude(gx, gy)
    
    gx_s = cv2.Scharr(srch_orig, cv2.CV_32F, 1, 0)
    gy_s = cv2.Scharr(srch_orig, cv2.CV_32F, 0, 1)
    srch_grad = cv2.magnitude(gx_s, gy_s)
    reps_ref.append(ref_grad)
    reps_srch.append(srch_grad)
    
    # 5. Mild Gaussian Blur
    ref_blur = cv2.GaussianBlur(ref_orig, (3, 3), 0.8)
    srch_blur = cv2.GaussianBlur(srch_orig, (3, 3), 0.8)
    reps_ref.append(ref_blur)
    reps_srch.append(srch_blur)
    
    # 6. Mild Sharpening
    ref_sharp = cv2.addWeighted(ref_orig, 1.5, ref_blur, -0.5, 0)
    srch_sharp = cv2.addWeighted(srch_orig, 1.5, srch_blur, -0.5, 0)
    reps_ref.append(ref_sharp)
    reps_srch.append(srch_sharp)
    
    rep_names = ['original', 'normalized', 'highpass', 'gradient', 'blur', 'sharp']
    return reps_ref, reps_srch, rep_names

def run_v37_stability_experiment(pairs_csv='data/phase2_dev/pairs.csv', output_csv='phase2/V37_LOCALIZATION/v37_results.csv'):
    df_pairs = pd.read_csv(pairs_csv)
    # Filter for present ground-truth pairs or all pairs
    # Present pairs have gt_found == 1
    present_pairs = df_pairs[df_pairs['gt_found'] == 1].copy()
    
    results = []
    print(f"Running V37 Representation Stability Experiment on {len(present_pairs)} present pairs...", flush=True)
    
    t0 = time.time()
    for idx, row in present_pairs.iterrows():
        pid = row['pair_id']
        if (len(results) + 1) % 5 == 0 or len(results) == 0:
            print(f"[{len(results)+1}/{len(present_pairs)}] Processing {pid}...", flush=True)
        gt_x = float(row['gt_x'])
        gt_y = float(row['gt_y'])
        
        ref_path = os.path.join('data/phase2_dev', row['reference_path'].replace('\\', '/'))
        srch_path = os.path.join('data/phase2_dev', row['search_path'].replace('\\', '/'))
        
        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(srch_path, cv2.IMREAD_GRAYSCALE)
        
        if ref_img is None or search_img is None:
            print(f"Warning: could not read images for {pid}")
            continue
            
        # 1. Base Pose Fallback
        pose = perform_pose_fallback_search(ref_img, search_img)
        corr_plane_orig = pose['corr_plane']
        best_template = pose['best_template']
        est_scale = pose['best_scale']
        est_theta = pose['best_theta']
        tw, th = best_template.shape[::-1]
        
        # 2. Candidate Extraction (V25 Baseline)
        cands = extract_candidates_akhilesh(corr_plane_orig, tw, th, ref_img, search_img, est_scale, est_theta, max_final_k=200)
        cands = cluster_replica_families(cands, est_scale)
        
        if len(cands) == 0:
            continue
            
        # For V37 analysis, evaluate the top candidate (V25 rank 1) or top 3 candidates
        # Focus on Candidate 0 (V25 primary candidate)
        c0 = cands[0]
        base_px, base_py = c0['peak_x'], c0['peak_y']
        base_cx, base_cy = c0['cx'], c0['cy']
        loc_err = float(np.hypot(base_cx - gt_x, base_cy - gt_y))
        status = "SAFE" if loc_err <= 5.0 else "WRONG"
        
        # 3. Generate representations
        reps_ref, reps_srch, rep_names = get_representations(ref_img, search_img)
        
        # Build rotated template for each representation
        # Rotation angle is est_theta
        h, w = ref_img.shape[:2]
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, est_theta, 1.0)
        
        rep_scores = {}
        rep_ranks = {}
        rep_coords = {}
        
        # Evaluate correlation plane under each representation
        for r_idx, r_name in enumerate(rep_names):
            r_ref = reps_ref[r_idx]
            r_srch = reps_srch[r_idx]
            
            # Crop/rotate template under representation
            # Crop center box based on scale
            crop_size = int(round(128 * (est_scale / 10.0)))
            crop_size = max(16, min(crop_size, min(h, w)))
            x1 = max(0, int(center[0] - crop_size // 2))
            y1 = max(0, int(center[1] - crop_size // 2))
            
            ref_crop = r_ref[y1:y1+crop_size, x1:x1+crop_size]
            rot_tpl = cv2.warpAffine(ref_crop, M, (crop_size, crop_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            
            # Match template
            corr_plane_r = cv2.matchTemplate(r_srch, rot_tpl, cv2.TM_CCOEFF_NORMED)
            ch_r, cw_r = corr_plane_r.shape[:2]
            
            # Local peak search around base_px, base_py (window +/- 3)
            win_y1, win_y2 = max(0, base_py - 3), min(ch_r, base_py + 4)
            win_x1, win_x2 = max(0, base_px - 3), min(cw_r, base_px + 4)
            
            sub_plane = corr_plane_r[win_y1:win_y2, win_x1:win_x2]
            if sub_plane.size > 0:
                _, max_val, _, max_loc = cv2.minMaxLoc(sub_plane)
                peak_x_r = win_x1 + max_loc[0]
                peak_y_r = win_y1 + max_loc[1]
                score_r = float(max_val)
            else:
                peak_x_r, peak_y_r = base_px, base_py
                score_r = float(corr_plane_r[min(base_py, ch_r-1), min(base_px, cw_r-1)])
                
            cx_r = peak_x_r + rot_tpl.shape[1] / 2.0
            cy_r = peak_y_r + rot_tpl.shape[0] / 2.0
            
            # Compute candidate rank in corr_plane_r among the pool of 200 V25 candidate positions
            cand_scores_r = []
            for cand in cands:
                c_px, c_py = cand['peak_x'], cand['peak_y']
                if 0 <= c_py < ch_r and 0 <= c_px < cw_r:
                    cand_scores_r.append(float(corr_plane_r[c_py, c_px]))
                else:
                    cand_scores_r.append(-1.0)
                    
            # Rank of candidate 0 (1-indexed)
            sorted_indices = np.argsort(cand_scores_r)[::-1]
            rank_r = int(np.where(sorted_indices == 0)[0][0]) + 1
            
            rep_scores[r_name] = score_r
            rep_ranks[r_name] = rank_r
            rep_coords[r_name] = (cx_r, cy_r)
            
        # Summary metrics
        scores_arr = np.array([rep_scores[rn] for rn in rep_names])
        ranks_arr = np.array([rep_ranks[rn] for rn in rep_names])
        
        score_mean = float(np.mean(scores_arr))
        score_std = float(np.std(scores_arr))
        
        rank_original = rep_ranks['original']
        rank_normalized = rep_ranks['normalized']
        rank_highpass = rep_ranks['highpass']
        rank_gradient = rep_ranks['gradient']
        rank_blur = rep_ranks['blur']
        rank_sharp = rep_ranks['sharp']
        rank_std = float(np.std(ranks_arr))
        
        # Coordinate shifts and std
        xs = np.array([rep_coords[rn][0] for rn in rep_names])
        ys = np.array([rep_coords[rn][1] for rn in rep_names])
        
        x_shift = float(np.mean(np.abs(xs - base_cx)))
        y_shift = float(np.mean(np.abs(ys - base_cy)))
        coord_std = float(np.sqrt(np.var(xs) + np.var(ys)))
        
        winner_frequency = float(np.mean(ranks_arr == 1))
        representation_agreement = int(np.sum(ranks_arr <= 3))
        
        rec = {
            'pair_id': pid,
            'candidate_rank': 1,
            'gt_x': gt_x,
            'gt_y': gt_y,
            'pred_x': base_cx,
            'pred_y': base_cy,
            'loc_error': loc_err,
            'status': status,
            
            'score_original': rep_scores['original'],
            'score_normalized': rep_scores['normalized'],
            'score_highpass': rep_scores['highpass'],
            'score_gradient': rep_scores['gradient'],
            'score_blur': rep_scores['blur'],
            'score_sharp': rep_scores['sharp'],
            
            'score_mean': score_mean,
            'score_std': score_std,
            
            'rank_original': rank_original,
            'rank_normalized': rank_normalized,
            'rank_highpass': rank_highpass,
            'rank_gradient': rank_gradient,
            'rank_blur': rank_blur,
            'rank_sharp': rank_sharp,
            'rank_std': rank_std,
            
            'x_shift': x_shift,
            'y_shift': y_shift,
            'coordinate_std': coord_std,
            'winner_frequency': winner_frequency,
            'representation_agreement': representation_agreement
        }
        results.append(rec)
        
    df_res = pd.DataFrame(results)
    df_res.to_csv(output_csv, index=False)
    elapsed = time.time() - t0
    print(f"V37 experiment complete in {elapsed:.2f}s. Results saved to {output_csv}.")
    
    # Generate V37_REPORT.md
    generate_v37_report(df_res, elapsed)
    return df_res

def generate_v37_report(df: pd.DataFrame, elapsed: float):
    safe_df = df[df['status'] == 'SAFE']
    wrong_df = df[df['status'] == 'WRONG']
    
    report_md = f"""# V37 Representation Stability Analysis Report

## Executive Summary
- **Total Pairs Analyzed**: {len(df)}
- **SAFE Candidates (loc_error <= 5.0px)**: {len(safe_df)} ({len(safe_df)/len(df)*100:.1f}%)
- **WRONG Candidates (loc_error > 5.0px)**: {len(wrong_df)} ({len(wrong_df)/len(df)*100:.1f}%)
- **Runtime**: {elapsed:.2f} seconds

---

## Stability Metrics: SAFE vs WRONG Candidates

| Metric | SAFE (n={len(safe_df)}) | WRONG (n={len(wrong_df)}) | Delta (SAFE - WRONG) |
|---|---|---|---|
| **Score Mean** | {safe_df['score_mean'].mean():.4f} | {wrong_df['score_mean'].mean():.4f} | {safe_df['score_mean'].mean() - wrong_df['score_mean'].mean():+.4f} |
| **Score Std** | {safe_df['score_std'].mean():.4f} | {wrong_df['score_std'].mean():.4f} | {safe_df['score_std'].mean() - wrong_df['score_std'].mean():+.4f} |
| **Rank Std** | {safe_df['rank_std'].mean():.4f} | {wrong_df['rank_std'].mean():.4f} | {safe_df['rank_std'].mean() - wrong_df['rank_std'].mean():+.4f} |
| **Coordinate Std (px)** | {safe_df['coordinate_std'].mean():.4f} | {wrong_df['coordinate_std'].mean():.4f} | {safe_df['coordinate_std'].mean() - wrong_df['coordinate_std'].mean():+.4f} |
| **Winner Frequency** | {safe_df['winner_frequency'].mean():.4f} | {wrong_df['winner_frequency'].mean():.4f} | {safe_df['winner_frequency'].mean() - wrong_df['winner_frequency'].mean():+.4f} |
| **Representation Agreement (<=3)** | {safe_df['representation_agreement'].mean():.2f} / 6 | {wrong_df['representation_agreement'].mean():.2f} / 6 | {safe_df['representation_agreement'].mean() - wrong_df['representation_agreement'].mean():+.2f} |

---

## Per-Representation Rank 1 Frequency
- **Original**: SAFE rank=1 in {safe_df['rank_original'].eq(1).mean()*100:.1f}% | WRONG rank=1 in {wrong_df['rank_original'].eq(1).mean()*100:.1f}%
- **Normalized**: SAFE rank=1 in {safe_df['rank_normalized'].eq(1).mean()*100:.1f}% | WRONG rank=1 in {wrong_df['rank_normalized'].eq(1).mean()*100:.1f}%
- **High-pass**: SAFE rank=1 in {safe_df['rank_highpass'].eq(1).mean()*100:.1f}% | WRONG rank=1 in {wrong_df['rank_highpass'].eq(1).mean()*100:.1f}%
- **Gradient**: SAFE rank=1 in {safe_df['rank_gradient'].eq(1).mean()*100:.1f}% | WRONG rank=1 in {wrong_df['rank_gradient'].eq(1).mean()*100:.1f}%
- **Blur**: SAFE rank=1 in {safe_df['rank_blur'].eq(1).mean()*100:.1f}% | WRONG rank=1 in {wrong_df['rank_blur'].eq(1).mean()*100:.1f}%
- **Sharp**: SAFE rank=1 in {safe_df['rank_sharp'].eq(1).mean()*100:.1f}% | WRONG rank=1 in {wrong_df['rank_sharp'].eq(1).mean()*100:.1f}%

---

## Key Findings & Signal Assessment
1. **Geometric Stability**:
   - SAFE candidates show geometric coordinate std of `{safe_df['coordinate_std'].mean():.3f}px` vs `{wrong_df['coordinate_std'].mean():.3f}px` for WRONG candidates.
2. **Representation Agreement**:
   - SAFE candidates remain top-ranked across `{safe_df['representation_agreement'].mean():.2f}` representations, compared to `{wrong_df['representation_agreement'].mean():.2f}` for WRONG candidates.

---
*Report generated automatically by `v37_stability.py`*
"""
    with open('phase2/V37_LOCALIZATION/V37_REPORT.md', 'w') as f:
        f.write(report_md)
    print("Saved V37_REPORT.md")

if __name__ == '__main__':
    run_v37_stability_experiment()
