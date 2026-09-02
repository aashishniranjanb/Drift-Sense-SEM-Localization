import pandas as pd
import numpy as np
import cv2
import sys
import os

sys.path.append("phase2")
from scale_search import coarse_to_fine_scale_search
from rotation_search import coarse_to_fine_rotation_search, rotate_image

def local_maxima_detector(corr, w=4):
    from scipy.ndimage import maximum_filter
    size = 2 * w + 1
    local_max = (maximum_filter(corr, size=size) == corr)
    local_max = local_max & (corr > 0.01)
    y_indices, x_indices = np.where(local_max)
    scores = corr[y_indices, x_indices]
    sorted_idx = np.argsort(scores)[::-1]
    return list(zip(x_indices[sorted_idx], y_indices[sorted_idx], scores[sorted_idx]))

def pose_hypothesis_retrieval(ref_img, search_img, num_hypotheses=3, extract_k=20, suppress_r=8):
    ref_f = ref_img.astype(np.float32)
    search_f = search_img.astype(np.float32)
    ref_h, ref_w = ref_f.shape[:2]
    
    # 1. Coarse Scale sweep
    scale_min, scale_max = 8.0, 12.0
    coarse_step = 0.5
    coarse_scales = np.arange(scale_min, scale_max + 1e-5, coarse_step)
    
    coarse_results = []
    for s in coarse_scales:
        tw = int(round(ref_w / s))
        th = int(round(ref_h / s))
        tpl = cv2.resize(ref_f, (tw, th), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(search_f, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        coarse_results.append((s, max_val))
        
    # Sort scale hypotheses by score descending
    coarse_results.sort(key=lambda x: x[1], reverse=True)
    top_hypotheses = coarse_results[:num_hypotheses]
    
    union_candidates = []
    
    for s_hyp, _ in top_hypotheses:
        # Run local fine scale sweep around this hypothesis
        fine_min = max(scale_min, s_hyp - 0.3)
        fine_max = min(scale_max, s_hyp + 0.3)
        fine_scales = np.arange(fine_min, fine_max + 1e-5, 0.1)
        
        best_fine_score = -1.0
        best_fine_scale = s_hyp
        best_fine_template = None
        
        for fs in fine_scales:
            tw = int(round(ref_w / fs))
            th = int(round(ref_h / fs))
            tpl = cv2.resize(ref_f, (tw, th), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(search_f, tpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val > best_fine_score:
                best_fine_score = max_val
                best_fine_scale = fs
                best_fine_template = tpl
                
        # Find rotation for this scale
        rot_res = coarse_to_fine_rotation_search(best_fine_template, search_img)
        corr = rot_res["corr_plane"]
        rotated_template = rot_res["rotated_template"]
        
        th, tw = rotated_template.shape[:2]
        ch, cw = corr.shape[:2]
        
        # Extract candidates from this correlation plane
        work = corr.copy()
        for _ in range(extract_k):
            _, max_val, _, max_loc = cv2.minMaxLoc(work)
            if max_val <= -1.0 or np.isnan(max_val): break
            px, py = max_loc
            cx, cy = px + tw / 2.0, py + th / 2.0
            
            union_candidates.append({
                "cx": cx,
                "cy": cy,
                "corr_score": float(max_val)
            })
            
            # Suppress
            y1, y2 = max(0, py - suppress_r), min(ch, py + suppress_r + 1)
            x1, x2 = max(0, px - suppress_r), min(cw, px + suppress_r + 1)
            work[y1:y2, x1:x2] = -999.0
            
    # Deduplicate union candidates (NMS radius = 3.0 px)
    union_candidates.sort(key=lambda x: x["corr_score"], reverse=True)
    unique_candidates = []
    for c in union_candidates:
        is_duplicate = False
        for u in unique_candidates:
            if np.hypot(c["cx"] - u["cx"], c["cy"] - u["cy"]) < 3.0:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_candidates.append(c)
            
    return unique_candidates

def run_hypothesis_sweep():
    df = pd.read_csv("data/phase2_dev/pairs.csv")
    present_df = df[df["gt_found"] == 1]
    total_present = len(present_df)
    
    print("Pre-loading images...")
    pairs_data = []
    for idx, r in present_df.iterrows():
        ref_img = cv2.imread("data/phase2_dev/" + r["reference_path"], cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread("data/phase2_dev/" + r["search_path"], cv2.IMREAD_GRAYSCALE)
        pairs_data.append((ref_img, search_img, r["gt_x"], r["gt_y"]))
        
    hyp_counts = [1, 3, 5]
    results = []
    
    for h_count in hyp_counts:
        hits = {20: 0, 50: 0, 100: 0}
        
        for ref_img, search_img, gt_x, gt_y in pairs_data:
            # We extract extract_k = 40 candidates from each hypothesis
            candidates = pose_hypothesis_retrieval(ref_img, search_img, num_hypotheses=h_count, extract_k=40)
            
            best_rank = None
            for rank, c in enumerate(candidates[:100]):
                if np.hypot(c["cx"] - gt_x, c["cy"] - gt_y) <= 5.0:
                    best_rank = rank
                    break
                    
            if best_rank is not None:
                if best_rank < 20: hits[20] += 1
                if best_rank < 50: hits[50] += 1
                if best_rank < 100: hits[100] += 1
                
        rec_20 = (hits[20] / total_present) * 100
        rec_50 = (hits[50] / total_present) * 100
        rec_100 = (hits[100] / total_present) * 100
        
        print(f"Pose Hypotheses={h_count} | Top-20={rec_20:.1f}%, Top-50={rec_50:.1f}%, Top-100={rec_100:.1f}%")
        
        results.append({
            "hypotheses_count": h_count,
            "top20_recall": rec_20,
            "top50_recall": rec_50,
            "top100_recall": rec_100
        })
        
    os.makedirs("results/phase2/V11_MAIN_TRACK", exist_ok=True)
    pd.DataFrame(results).to_csv("results/phase2/V11_MAIN_TRACK/V11_POSE_HYPOTHESIS_SWEEP.csv", index=False)
    print("Report written to results/phase2/V11_MAIN_TRACK/V11_POSE_HYPOTHESIS_SWEEP.csv")

if __name__ == "__main__":
    run_hypothesis_sweep()
