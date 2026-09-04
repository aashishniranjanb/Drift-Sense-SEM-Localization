import os
import sys
import cv2
import numpy as np
import pandas as pd

sys.path.append("phase2")
from rotation_search import coarse_to_fine_rotation_search

def coarse_scale_search_custom(ref_img, search_img, step=0.5):
    ref_f = ref_img.astype(np.float32)
    search_f = search_img.astype(np.float32)
    ref_h, ref_w = ref_f.shape[:2]
    
    scales = np.arange(8.0, 12.01, step)
    best_score = -1.0
    best_scale = 10.0
    best_tpl = None
    
    for s in scales:
        tw = int(round(ref_w / s))
        th = int(round(ref_h / s))
        tpl = cv2.resize(ref_f, (tw, th), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(search_f, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val > best_score:
            best_score = max_val
            best_scale = s
            best_tpl = tpl
            
    # Fine refinement
    fine_scales = np.arange(max(8.0, best_scale - 0.25), min(12.0, best_scale + 0.26), 0.05)
    for fs in fine_scales:
        tw = int(round(ref_w / fs))
        th = int(round(ref_h / fs))
        tpl = cv2.resize(ref_f, (tw, th), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(search_f, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val > best_score:
            best_score = max_val
            best_scale = fs
            best_tpl = tpl
            
    return {"best_scale": best_scale, "best_template": best_tpl}

def evaluate_scale_resolution():
    csv_path = "data/phase2_dev/pairs.csv"
    df = pd.read_csv(csv_path)
    present_df = df[df["gt_found"] == 1].copy()
    
    print("Evaluating scale resolutions on 140 present pairs...")
    
    steps = [0.5, 0.2, 0.1]
    
    for step in steps:
        hits = {20: 0, 50: 0, 100: 0}
        scale_errors = []
        
        for idx, r in present_df.iterrows():
            ref_img = cv2.imread(os.path.join("data/phase2_dev", r["reference_path"]), cv2.IMREAD_GRAYSCALE)
            search_img = cv2.imread(os.path.join("data/phase2_dev", r["search_path"]), cv2.IMREAD_GRAYSCALE)
            
            gt_x = float(r["gt_x"])
            gt_y = float(r["gt_y"])
            gt_scale = float(r.get("gt_scale", 10.0)) if "gt_scale" in r else 10.0
            
            scale_res = coarse_scale_search_custom(ref_img, search_img, step=step)
            scale_errors.append(abs(scale_res["best_scale"] - gt_scale))
            
            rot_res = coarse_to_fine_rotation_search(scale_res["best_template"], search_img)
            corr = rot_res["corr_plane"]
            th, tw = rot_res["rotated_template"].shape[:2]
            ch, cw = corr.shape[:2]
            
            # Extract NMS r=5
            work = corr.copy()
            cands = []
            for _ in range(100):
                _, max_val, _, max_loc = cv2.minMaxLoc(work)
                if max_val <= 0.05 or np.isnan(max_val): break
                px, py = max_loc
                cx, cy = px + tw/2.0, py + th/2.0
                cands.append({"cx": cx, "cy": cy})
                y1, y2 = max(0, py - 5), min(ch, py + 6)
                x1, x2 = max(0, px - 5), min(cw, px + 6)
                work[y1:y2, x1:x2] = -999.0
                
            best_rank = None
            for rank, c in enumerate(cands):
                if np.hypot(c["cx"] - gt_x, c["cy"] - gt_y) <= 5.0:
                    best_rank = rank
                    break
            if best_rank is not None:
                if best_rank < 20: hits[20] += 1
                if best_rank < 50: hits[50] += 1
                if best_rank < 100: hits[100] += 1
                
        n = len(present_df)
        print(f"Coarse Step: {step:.1f} | Mean Scale Error: {np.mean(scale_errors):.4f} | Top-20: {hits[20]/n*100:.1f}%, Top-50: {hits[50]/n*100:.1f}%, Top-100: {hits[100]/n*100:.1f}%")

if __name__ == "__main__":
    evaluate_scale_resolution()
