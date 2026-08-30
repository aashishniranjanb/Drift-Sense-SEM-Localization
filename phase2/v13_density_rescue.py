import os
import sys
import time
import cv2
import numpy as np
import pandas as pd

sys.path.append("phase2")
from scale_search import coarse_to_fine_scale_search
from rotation_search import coarse_to_fine_rotation_search, rotate_image

def load_dev_data():
    csv_path = "data/phase2_dev/pairs.csv"
    df = pd.read_csv(csv_path)
    present_df = df[df["gt_found"] == 1].copy()
    
    pairs = []
    for _, row in present_df.iterrows():
        ref_path = os.path.join("data/phase2_dev", row["reference_path"])
        search_path = os.path.join("data/phase2_dev", row["search_path"])
        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
        pairs.append({
            "pair_id": row["pair_id"],
            "ref_img": ref_img,
            "search_img": search_img,
            "gt_x": float(row["gt_x"]),
            "gt_y": float(row["gt_y"]),
            "set_type": row["set_type"]
        })
    return pairs

def evaluate_retrieval(candidates, gt_x, gt_y, thresholds=(20, 50, 100)):
    results = {k: 0 for k in thresholds}
    best_rank = None
    for rank, c in enumerate(candidates):
        dist = np.hypot(c["cx"] - gt_x, c["cy"] - gt_y)
        if dist <= 5.0 and best_rank is None:
            best_rank = rank
            break
    if best_rank is not None:
        for k in thresholds:
            if best_rank < k:
                results[k] = 1
    return results

# Experiment 1: Region-Partitioned Local Quota Peak Extraction
def extract_quota_candidates(corr, tw, th, grid_size=(3, 3), per_cell=5, global_k=50, r=5):
    ch, cw = corr.shape
    candidates = []
    
    # Global extraction
    work = corr.copy()
    for _ in range(global_k):
        _, max_val, _, max_loc = cv2.minMaxLoc(work)
        if max_val <= 0.05 or np.isnan(max_val): break
        px, py = max_loc
        candidates.append({"px": px, "py": py, "score": max_val, "source": "global"})
        y1, y2 = max(0, py - r), min(ch, py + r + 1)
        x1, x2 = max(0, px - r), min(cw, px + r + 1)
        work[y1:y2, x1:x2] = -999.0
        
    # Local quota extraction
    gh, gw = grid_size
    cell_h, cell_w = ch / gh, cw / gw
    for gi in range(gh):
        for gj in range(gw):
            y1_cell = int(gi * cell_h)
            y2_cell = int((gi + 1) * cell_h) if gi < gh - 1 else ch
            x1_cell = int(gj * cell_w)
            x2_cell = int((gj + 1) * cell_w) if gj < gw - 1 else cw
            
            sub = corr[y1_cell:y2_cell, x1_cell:x2_cell].copy()
            sch, scw = sub.shape
            for _ in range(per_cell):
                _, max_val, _, max_loc = cv2.minMaxLoc(sub)
                if max_val <= 0.05 or np.isnan(max_val): break
                spx, spy = max_loc
                candidates.append({"px": x1_cell + spx, "py": y1_cell + spy, "score": max_val, "source": f"local_{gi}_{gj}"})
                y1_sub, y2_sub = max(0, spy - r), min(sch, spy + r + 1)
                x1_sub, x2_sub = max(0, spx - r), min(scw, spx + r + 1)
                sub[y1_sub:y2_sub, x1_sub:x2_sub] = -999.0
                
    # Deduplicate
    candidates = sorted(candidates, key=lambda c: c["score"], reverse=True)
    unique = []
    for c in candidates:
        cx = c["px"] + tw / 2.0
        cy = c["py"] + th / 2.0
        if not any(np.hypot(cx - u["cx"], cy - u["cy"]) < r for u in unique):
            unique.append({"cx": cx, "cy": cy, "score": c["score"]})
            if len(unique) >= 100: break
    return unique

# Experiment 2: Spatially Diverse Secondary Queue Rescue
def extract_diverse_rescue_candidates(corr, tw, th, primary_ratio=0.7, r=5):
    ch, cw = corr.shape
    primary_q = []
    secondary_q = []
    
    work = corr.copy()
    for _ in range(150):
        _, max_val, _, max_loc = cv2.minMaxLoc(work)
        if max_val <= 0.05 or np.isnan(max_val): break
        px, py = max_loc
        cx = px + tw / 2.0
        cy = py + th / 2.0
        
        # Check if too close to existing primary candidates
        is_close = any(np.hypot(cx - p["cx"], cy - p["cy"]) < 15.0 for p in primary_q)
        cand = {"cx": cx, "cy": cy, "score": max_val}
        
        if not is_close:
            primary_q.append(cand)
        else:
            secondary_q.append(cand)
            
        y1, y2 = max(0, py - r), min(ch, py + r + 1)
        x1, x2 = max(0, px - r), min(cw, px + r + 1)
        work[y1:y2, x1:x2] = -999.0
        
    # Construct pool mixing primary and secondary
    max_primary = int(100 * primary_ratio)
    max_secondary = 100 - max_primary
    
    pool = primary_q[:max_primary] + secondary_q[:max_secondary]
    pool = sorted(pool, key=lambda x: x["score"], reverse=True)
    return pool

# Experiment 3: Pose-Normalized Hypothesis Fusion (Z-Score & Rank normalization)
def pose_normalized_retrieval(ref_img, search_img, H=3, normalization="z-score"):
    ref_f = ref_img.astype(np.float32)
    search_f = search_img.astype(np.float32)
    ref_h, ref_w = ref_f.shape[:2]
    
    # 1. Coarse Scale sweep
    coarse_scales = np.arange(8.0, 12.01, 0.5)
    coarse_scores = []
    for s in coarse_scales:
        tw = int(round(ref_w / s))
        th = int(round(ref_h / s))
        tpl = cv2.resize(ref_f, (tw, th), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(search_f, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        coarse_scores.append((s, max_val))
    coarse_scores.sort(key=lambda x: x[1], reverse=True)
    top_scales = coarse_scores[:H]
    
    union_candidates = []
    final_tw, final_th = None, None
    
    for rank_idx, (s_hyp, _) in enumerate(top_scales):
        scale_res = coarse_to_fine_scale_search(ref_img, search_img, scale_min=max(8.0, s_hyp-0.3), scale_max=min(12.0, s_hyp+0.3))
        rot_res = coarse_to_fine_rotation_search(scale_res["best_template"], search_img)
        corr = rot_res["corr_plane"]
        tw, th = rot_res["rotated_template"].shape[1], rot_res["rotated_template"].shape[0]
        final_tw, final_th = tw, th
        
        # Calculate statistics for normalization
        mu = np.mean(corr)
        sigma = np.std(corr)
        
        cands = []
        work = corr.copy()
        ch, cw = work.shape
        for r_rank in range(50):
            _, max_val, _, max_loc = cv2.minMaxLoc(work)
            if max_val <= 0.05 or np.isnan(max_val): break
            px, py = max_loc
            
            # Normalize
            if normalization == "z-score":
                norm_score = (max_val - mu) / (sigma + 1e-6)
            elif normalization == "rank":
                norm_score = 50 - r_rank
            else:  # raw
                norm_score = max_val
                
            cands.append({
                "px": px,
                "py": py,
                "cx": px + tw / 2.0,
                "cy": py + th / 2.0,
                "score": norm_score
            })
            y1, y2 = max(0, py - 5), min(ch, py + 6)
            x1, x2 = max(0, px - 5), min(cw, px + 6)
            work[y1:y2, x1:x2] = -999.0
        union_candidates.extend(cands)
        
    # Deduplicate
    union_candidates = sorted(union_candidates, key=lambda x: x["score"], reverse=True)
    unique = []
    for c in union_candidates:
        if not any(np.hypot(c["cx"] - u["cx"], c["cy"] - u["cy"]) < 5.0 for u in unique):
            unique.append(c)
            if len(unique) >= 100: break
    return unique

def main():
    pairs = load_dev_data()
    n = len(pairs)
    
    # Precompute corr planes for Exp 1 & 2
    cached = []
    print("Pre-computing correlation planes...")
    for p in pairs:
        scale_res = coarse_to_fine_scale_search(p["ref_img"], p["search_img"])
        rot_res = coarse_to_fine_rotation_search(scale_res["best_template"], p["search_img"])
        cached.append({
            "corr": rot_res["corr_plane"],
            "tw": rot_res["rotated_template"].shape[1],
            "th": rot_res["rotated_template"].shape[0],
            "gt_x": p["gt_x"],
            "gt_y": p["gt_y"]
        })
        
    # ----------------------------------------------------
    # Experiment 1: Quota / Grid Extractor
    # ----------------------------------------------------
    print("\n--- [Exp 1] Testing Region-Partitioned Quota Extractor ---")
    grid_sizes = [(2,2), (3,3), (4,4)]
    for g in grid_sizes:
        hits = {20: 0, 50: 0, 100: 0}
        for item in cached:
            cands = extract_quota_candidates(item["corr"], item["tw"], item["th"], grid_size=g, per_cell=5, global_k=50)
            res = evaluate_retrieval(cands, item["gt_x"], item["gt_y"])
            for k in [20, 50, 100]: hits[k] += res[k]
        print(f"Grid {g[0]}x{g[1]} -> Top-20: {hits[20]/n*100:.2f}%, Top-50: {hits[50]/n*100:.2f}%, Top-100: {hits[100]/n*100:.2f}%")
        
    # ----------------------------------------------------
    # Experiment 2: Spatial Diverse Sweep (Primary/Secondary Ratio)
    # ----------------------------------------------------
    print("\n--- [Exp 2] Testing Spatial Diverse Sweep (Ratio Sweep) ---")
    ratios = [0.9, 0.8, 0.7, 0.6, 0.5]
    for r in ratios:
        hits = {20: 0, 50: 0, 100: 0}
        for item in cached:
            cands = extract_diverse_rescue_candidates(item["corr"], item["tw"], item["th"], primary_ratio=r)
            res = evaluate_retrieval(cands, item["gt_x"], item["gt_y"])
            for k in [20, 50, 100]: hits[k] += res[k]
        print(f"Ratio {int(r*100)}/{int((1-r)*100)} -> Top-20: {hits[20]/n*100:.2f}%, Top-50: {hits[50]/n*100:.2f}%, Top-100: {hits[100]/n*100:.2f}%")
        
    # ----------------------------------------------------
    # Experiment 3: Pose-Normalized Hypothesis Fusion
    # ----------------------------------------------------
    print("\n--- [Exp 3] Testing Pose-Normalized Hypothesis Fusion ---")
    norm_methods = ["raw", "z-score", "rank"]
    for norm in norm_methods:
        hits = {20: 0, 50: 0, 100: 0}
        for p in pairs:
            cands = pose_normalized_retrieval(p["ref_img"], p["search_img"], H=3, normalization=norm)
            res = evaluate_retrieval(cands, p["gt_x"], p["gt_y"])
            for k in [20, 50, 100]: hits[k] += res[k]
        print(f"Normalization: {norm:<10} -> Top-20: {hits[20]/n*100:.2f}%, Top-50: {hits[50]/n*100:.2f}%, Top-100: {hits[100]/n*100:.2f}%")

if __name__ == "__main__":
    main()
