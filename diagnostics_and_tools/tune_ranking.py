import pandas as pd
import numpy as np
import cv2

from inference import (
    extract_distinct_spatial_peaks,
    extract_gradient_map,
    normalize_intensity,
    local_phase_correlation,
    subpixel_refine_2d
)

df = pd.read_csv('data/benchmark_120/manifest.csv')

print("Testing full scale bank (0.95-1.05) + rotation bank (-3 to +3 deg)...")

def evaluate_full(scales, rotations, w_raw=0.55, w_grad=0.30, w_phase=0.15):
    correct_1px, correct_3px, correct_5px = 0, 0, 0
    errors = []
    diff_accs = {d: 0 for d in ["Easy", "Medium", "Hard", "Adversarial"]}
    diff_totals = {d: 0 for d in ["Easy", "Medium", "Hard", "Adversarial"]}
    
    for idx, row in df.iterrows():
        ref_img = cv2.imread(row['reference_path'], cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(row['search_path'], cv2.IMREAD_GRAYSCALE)
        gt_x, gt_y = float(row['gt_x']), float(row['gt_y'])
        diff = row['difficulty']
        diff_totals[diff] += 1
        sh, sw = search_img.shape
        
        search_proc = cv2.GaussianBlur(search_img, (3, 3), 0.5)
        ref_proc = cv2.GaussianBlur(ref_img, (3, 3), 0.5)
            
        search_norm = normalize_intensity(search_proc)
        search_grad = extract_gradient_map(search_proc)
        
        all_cands = []
        for s in scales:
            tw = max(10, int(round(100 * s)))
            th = max(10, int(round(100 * s)))
            ref_s = cv2.resize(ref_proc, (tw, th), interpolation=cv2.INTER_AREA)
            
            for r in rotations:
                if abs(r) > 0.01:
                    M = cv2.getRotationMatrix2D((tw/2.0, th/2.0), r, 1.0)
                    ref_r = cv2.warpAffine(ref_s, M, (tw, th), borderMode=cv2.BORDER_REFLECT)
                else:
                    ref_r = ref_s
                    
                ref_n = normalize_intensity(ref_r)
                ref_g = extract_gradient_map(ref_r)
                
                c_i = cv2.matchTemplate(search_norm, ref_n, cv2.TM_CCOEFF_NORMED)
                c_g = cv2.matchTemplate(search_grad, ref_g, cv2.TM_CCOEFF_NORMED)
                c_combo = 0.55 * c_i + 0.45 * c_g
                
                peaks = extract_distinct_spatial_peaks(c_combo, top_k=4, min_distance=15)
                for p in peaks:
                    all_cands.append({
                        "x": p["x"] + tw / 2.0,
                        "y": p["y"] + th / 2.0,
                        "peak_x": p["x"],
                        "peak_y": p["y"],
                        "tw": tw,
                        "th": th,
                        "scale": s,
                        "rot": r,
                        "score": p["score"],
                        "corr_plane": c_combo,
                        "ref_n": ref_n,
                        "ref_g": ref_g
                    })
                    
        all_cands.sort(key=lambda c: c["score"], reverse=True)
        unique_cands = []
        for c in all_cands:
            if not any(np.hypot(c["x"] - u["x"], c["y"] - u["y"]) < 12 for u in unique_cands):
                unique_cands.append(c)
            if len(unique_cands) >= 15:
                break
                
        # Phase correlation & gradient verification
        for c in unique_cands:
            cx, cy = c["x"], c["y"]
            tw, th = c["tw"], c["th"]
            y1, y2 = max(0, int(round(cy - th/2.0))), min(sh, int(round(cy + th/2.0)))
            x1, x2 = max(0, int(round(cx - tw/2.0))), min(sw, int(round(cx + tw/2.0)))
            sp_n = search_norm[y1:y2, x1:x2]
            sp_g = search_grad[y1:y2, x1:x2]
            if sp_n.shape != (th, tw):
                sp_n = cv2.resize(sp_n, (tw, th))
                sp_g = cv2.resize(sp_g, (tw, th))
            dx, dy, ps = local_phase_correlation(c["ref_n"], sp_n)
            
            # Gradient correlation
            g_corr = float(np.corrcoef(c["ref_g"].ravel(), sp_g.ravel())[0, 1])
            if np.isnan(g_corr):
                g_corr = 0.0
                
            c["final_score"] = w_raw * c["score"] + w_grad * g_corr + w_phase * max(0.0, ps)
            
        unique_cands.sort(key=lambda c: c["final_score"], reverse=True)
        best = unique_cands[0]
        
        sub_x, sub_y = subpixel_refine_2d(best["corr_plane"], best["peak_x"], best["peak_y"])
        pred_x = sub_x + best["tw"] / 2.0
        pred_y = sub_y + best["th"] / 2.0
        
        err = float(np.hypot(pred_x - gt_x, pred_y - gt_y))
        errors.append(err)
        if err <= 1.0:
            correct_1px += 1
        if err <= 3.0:
            correct_3px += 1
        if err <= 5.0:
            correct_5px += 1
            diff_accs[diff] += 1
            
    err_arr = np.array(errors)
    res = {
        "acc1": round(correct_1px / 1.2, 2),
        "acc3": round(correct_3px / 1.2, 2),
        "acc5": round(correct_5px / 1.2, 2),
        "mean_err": round(float(np.mean(err_arr)), 2),
        "med_err": round(float(np.median(err_arr)), 2),
        "p95_err": round(float(np.percentile(err_arr, 95)), 2),
        "diff_breakdown": {d: round(diff_accs[d] / diff_totals[d] * 100, 1) for d in diff_accs}
    }
    return res

scales = [0.95, 0.98, 1.00, 1.02, 1.05]
rotations = [-3.0, -1.5, 0.0, 1.5, 3.0]
res = evaluate_full(scales, rotations)
print("\nFinal Results across 120 samples:")
print(f"Accuracy <= 1px: {res['acc1']}%")
print(f"Accuracy <= 3px: {res['acc3']}%")
print(f"Accuracy <= 5px: {res['acc5']}%")
print(f"Mean Error: {res['mean_err']} px | Median Error: {res['med_err']} px | P95 Error: {res['p95_err']} px")
print(f"Breakdown: {res['diff_breakdown']}")
