import os
import sys
import cv2
import numpy as np
import pandas as pd

sys.path.append("phase2")
from scale_search import coarse_to_fine_scale_search
from rotation_search import rotate_image

def compute_psr(corr, px, py, radius=10):
    ch, cw = corr.shape
    y1, y2 = max(0, py - radius), min(ch, py + radius + 1)
    x1, x2 = max(0, px - radius), min(cw, px + radius + 1)
    
    peak_val = corr[py, px]
    patch = corr[y1:y2, x1:x2].copy()
    
    # Mask out central 5x5
    cy, cx = py - y1, px - x1
    patch[max(0, cy - 2):min(patch.shape[0], cy + 3), max(0, cx - 2):min(patch.shape[1], cx + 3)] = np.nan
    
    valid = patch[~np.isnan(patch)]
    if len(valid) == 0:
        return 0.0
    mean_val = np.mean(valid)
    std_val = np.std(valid)
    if std_val < 1e-6:
        return 0.0
    return float((peak_val - mean_val) / std_val)

def evaluate_rotation_metrics():
    csv_path = "data/phase2_dev/pairs.csv"
    df = pd.read_csv(csv_path)
    present_df = df[df["gt_found"] == 1].copy()
    
    print("Evaluating rotation criteria on 140 present pairs...")
    
    coarse_angles = [-5.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0]
    
    methods = ["max_pixel", "psr", "top3_mean", "center_prior_weighted"]
    hits = {m: 0 for m in methods}
    rot_errors = {m: [] for m in methods}
    
    for idx, r in present_df.iterrows():
        ref_img = cv2.imread(os.path.join("data/phase2_dev", r["reference_path"]), cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(os.path.join("data/phase2_dev", r["search_path"]), cv2.IMREAD_GRAYSCALE)
        
        gt_x = float(r["gt_x"])
        gt_y = float(r["gt_y"])
        gt_rot = float(r.get("gt_rotation", 0.0)) if "gt_rotation" in r else 0.0
        
        scale_res = coarse_to_fine_scale_search(ref_img, search_img)
        tpl = scale_res["best_template"].astype(np.float32)
        search_f = search_img.astype(np.float32)
        sh, sw = search_f.shape[:2]
        
        # Test each coarse angle and collect correlation planes
        corr_planes = {}
        for theta in coarse_angles:
            rot_tpl = rotate_image(tpl, theta)
            res = cv2.matchTemplate(search_f, rot_tpl, cv2.TM_CCOEFF_NORMED)
            corr_planes[theta] = (res, rot_tpl)
            
        # 1. Method: max_pixel
        best_theta_max = max(coarse_angles, key=lambda th: np.max(corr_planes[th][0]))
        
        # 2. Method: PSR
        def get_psr_score(th):
            corr, _ = corr_planes[th]
            _, max_v, _, (px, py) = cv2.minMaxLoc(corr)
            return compute_psr(corr, px, py)
        best_theta_psr = max(coarse_angles, key=get_psr_score)
        
        # 3. Method: top3_mean
        def get_top3_score(th):
            corr, _ = corr_planes[th]
            flat = np.sort(corr.ravel())[::-1]
            return np.mean(flat[:3])
        best_theta_top3 = max(coarse_angles, key=get_top3_score)
        
        # 4. Method: center_prior_weighted
        def get_center_score(th):
            corr, rot_t = corr_planes[th]
            _, max_v, _, (px, py) = cv2.minMaxLoc(corr)
            cx, cy = px + rot_t.shape[1]/2.0, py + rot_t.shape[0]/2.0
            dist_to_center = np.hypot(cx - sw/2.0, cy - sh/2.0)
            return max_v * np.exp(-0.5 * (dist_to_center / 350.0)**2)
        best_theta_center = max(coarse_angles, key=get_center_score)
        
        selected_thetas = {
            "max_pixel": best_theta_max,
            "psr": best_theta_psr,
            "top3_mean": best_theta_top3,
            "center_prior_weighted": best_theta_center
        }
        
        # Check Top-100 recall for each method
        for m, theta in selected_thetas.items():
            rot_errors[m].append(abs(theta - gt_rot))
            corr, rot_t = corr_planes[theta]
            th_h, th_w = rot_t.shape[:2]
            
            # Extract NMS r=5
            work = corr.copy()
            ch, cw = work.shape
            cands = []
            for _ in range(100):
                _, max_v, _, (px, py) = cv2.minMaxLoc(work)
                if max_v <= 0.01 or np.isnan(max_v): break
                cx, cy = px + th_w/2.0, py + th_h/2.0
                cands.append((cx, cy))
                y1, y2 = max(0, py - 5), min(ch, py + 6)
                x1, x2 = max(0, px - 5), min(cw, px + 6)
                work[y1:y2, x1:x2] = -999.0
                
            hit = any(np.hypot(cx - gt_x, cy - gt_y) <= 5.0 for cx, cy in cands)
            if hit:
                hits[m] += 1
                
    n = len(present_df)
    print("\n=======================================================")
    print("         ROTATION SELECTION CRITERIA COMPARISON        ")
    print("=======================================================")
    for m in methods:
        mae = np.mean(rot_errors[m])
        rec = hits[m] / n * 100
        print(f"Method: {m:<22} | Mean Rot Error: {mae:.2f}° | Top-100 Recall: {hits[m]}/{n} ({rec:.2f}%)")
    print("=======================================================")

if __name__ == "__main__":
    evaluate_rotation_metrics()
