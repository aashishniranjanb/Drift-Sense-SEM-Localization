import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter

sys.path.append("phase2")
from scale_search import coarse_to_fine_scale_search
from rotation_search import coarse_to_fine_rotation_search, rotate_image
from channel_consensus import extract_gradient

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
            "gt_scale": float(row.get("gt_scale", 10.0)) if "gt_scale" in row else 10.0,
            "gt_rotation": float(row.get("gt_rotation", 0.0)) if "gt_rotation" in row else 0.0,
            "set_type": row["set_type"]
        })
    return pairs

def extract_spatial_grid_candidates(corr_plane, tw, th, grid_size=(3, 3), per_cell_k=15, global_k=40, nms_r=5, max_k=100):
    ch, cw = corr_plane.shape
    candidates = []
    
    # 1. Global NMS peaks
    work = corr_plane.copy()
    for _ in range(global_k):
        _, max_val, _, max_loc = cv2.minMaxLoc(work)
        if max_val <= 0.01 or np.isnan(max_val):
            break
        px, py = max_loc
        candidates.append({"px": int(px), "py": int(py), "score": float(max_val), "source": "global"})
        y1, y2 = max(0, py - nms_r), min(ch, py + nms_r + 1)
        x1, x2 = max(0, px - nms_r), min(cw, px + nms_r + 1)
        work[y1:y2, x1:x2] = -999.0
        
    # 2. Grid-based regional NMS peaks (prevents one corner from monopolizing the pool)
    gh, gw = grid_size
    cell_h = ch / gh
    cell_w = cw / gw
    
    for gi in range(gh):
        for gj in range(gw):
            y_start = int(gi * cell_h)
            y_end = int((gi + 1) * cell_h) if gi < gh - 1 else ch
            x_start = int(gj * cell_w)
            x_end = int((gj + 1) * cell_w) if gj < gw - 1 else cw
            
            sub_corr = corr_plane[y_start:y_end, x_start:x_end].copy()
            sch, scw = sub_corr.shape
            for _ in range(per_cell_k):
                _, max_val, _, max_loc = cv2.minMaxLoc(sub_corr)
                if max_val <= 0.01 or np.isnan(max_val):
                    break
                spx, spy = max_loc
                gpx = x_start + spx
                gpy = y_start + spy
                candidates.append({"px": int(gpx), "py": int(gpy), "score": float(max_val), "source": f"grid_{gi}_{gj}"})
                sy1, sy2 = max(0, spy - nms_r), min(sch, spy + nms_r + 1)
                sx1, sx2 = max(0, spx - nms_r), min(scw, spx + nms_r + 1)
                sub_corr[sy1:sy2, sx1:sx2] = -999.0
                
    # 3. Deduplicate
    candidates = sorted(candidates, key=lambda c: c["score"], reverse=True)
    unique = []
    for c in candidates:
        cx = c["px"] + tw / 2.0
        cy = c["py"] + th / 2.0
        is_dup = False
        for u in unique:
            if np.hypot(cx - u["cx"], cy - u["cy"]) < float(nms_r):
                is_dup = True
                break
        if not is_dup:
            unique.append({
                "cx": cx,
                "cy": cy,
                "score": c["score"],
                "source": c["source"]
            })
            if len(unique) >= max_k:
                break
    return unique

def evaluate_grid_extraction(pairs):
    print("\n--- Testing Spatial-Grid Partitioned Candidate Extraction ---")
    grid_configs = [
        {"name": "Pure Global NMS r=5", "grid": (1, 1), "per_cell": 0, "global_k": 100, "r": 5},
        {"name": "Pure Global NMS r=7", "grid": (1, 1), "per_cell": 0, "global_k": 100, "r": 7},
        {"name": "Pure Global NMS r=10", "grid": (1, 1), "per_cell": 0, "global_k": 100, "r": 10},
        {"name": "Grid 2x2 (per_cell=15, global=40, r=5)", "grid": (2, 2), "per_cell": 15, "global_k": 40, "r": 5},
        {"name": "Grid 3x3 (per_cell=10, global=30, r=5)", "grid": (3, 3), "per_cell": 10, "global_k": 30, "r": 5},
        {"name": "Grid 3x3 (per_cell=10, global=30, r=7)", "grid": (3, 3), "per_cell": 10, "global_k": 30, "r": 7},
        {"name": "Grid 4x4 (per_cell=5, global=30, r=5)", "grid": (4, 4), "per_cell": 5, "global_k": 30, "r": 5},
    ]
    
    # Precompute nominal scale/rot
    corr_planes = []
    for p in pairs:
        scale_res = coarse_to_fine_scale_search(p["ref_img"], p["search_img"])
        rot_res = coarse_to_fine_rotation_search(scale_res["best_template"], p["search_img"])
        corr_planes.append({
            "corr": rot_res["corr_plane"],
            "tw": rot_res["rotated_template"].shape[1],
            "th": rot_res["rotated_template"].shape[0],
            "gt_x": p["gt_x"],
            "gt_y": p["gt_y"]
        })
        
    for cfg in grid_configs:
        tot_hits = {20: 0, 50: 0, 100: 0}
        for item in corr_planes:
            cands = extract_spatial_grid_candidates(
                item["corr"], item["tw"], item["th"],
                grid_size=cfg["grid"],
                per_cell_k=cfg["per_cell"],
                global_k=cfg["global_k"],
                nms_r=cfg["r"],
                max_k=100
            )
            
            best_rank = None
            for rank, c in enumerate(cands):
                if np.hypot(c["cx"] - item["gt_x"], c["cy"] - item["gt_y"]) <= 5.0:
                    best_rank = rank
                    break
            if best_rank is not None:
                if best_rank < 20: tot_hits[20] += 1
                if best_rank < 50: tot_hits[50] += 1
                if best_rank < 100: tot_hits[100] += 1
                
        n = len(corr_planes)
        print(f"{cfg['name']:<42} -> Top-20: {tot_hits[20]/n*100:5.1f}%, Top-50: {tot_hits[50]/n*100:5.1f}%, Top-100: {tot_hits[100]/n*100:5.1f}%")

def evaluate_scale_search_enhancement(pairs):
    print("\n--- Investigating Scale Search Failure Cases ---")
    # Let's check how many cases have GT in corr plane if we use GT scale/rotation vs estimated
    errors_scale = []
    recovered_with_gt_pose = 0
    recovered_with_estimated_pose = 0
    
    for p in pairs:
        ref_f = p["ref_img"].astype(np.float32)
        search_f = p["search_img"].astype(np.float32)
        ref_h, ref_w = ref_f.shape[:2]
        
        # 1. Estimated scale
        scale_res = coarse_to_fine_scale_search(p["ref_img"], p["search_img"])
        rot_res = coarse_to_fine_rotation_search(scale_res["best_template"], p["search_img"])
        corr_est = rot_res["corr_plane"]
        rot_tpl = rot_res["rotated_template"]
        th, tw = rot_tpl.shape[:2]
        
        cands_est = extract_spatial_grid_candidates(corr_est, tw, th, grid_size=(3,3), per_cell_k=10, global_k=30, nms_r=5, max_k=100)
        hit_est = any(np.hypot(c["cx"] - p["gt_x"], c["cy"] - p["gt_y"]) <= 5.0 for c in cands_est)
        if hit_est:
            recovered_with_estimated_pose += 1
            
        # 2. Let's test a wider/denser scale search (step=0.2 instead of 0.5)
        # and test if that rescues scale mismatch cases
        dense_scales = np.arange(8.0, 12.01, 0.2)
        best_dense_val = -1.0
        best_dense_tpl = None
        for s in dense_scales:
            dtw = int(round(ref_w / s))
            dth = int(round(ref_h / s))
            dtpl = cv2.resize(ref_f, (dtw, dth), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(search_f, dtpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val > best_dense_val:
                best_dense_val = max_val
                best_dense_tpl = dtpl
                
        rot_dense = coarse_to_fine_rotation_search(best_dense_tpl, p["search_img"])
        corr_dense = rot_dense["corr_plane"]
        rth, rtw = rot_dense["rotated_template"].shape[:2]
        cands_dense = extract_spatial_grid_candidates(corr_dense, rtw, rth, grid_size=(3,3), per_cell_k=10, global_k=30, nms_r=5, max_k=100)
        hit_dense = any(np.hypot(c["cx"] - p["gt_x"], c["cy"] - p["gt_y"]) <= 5.0 for c in cands_dense)
        if hit_dense:
            recovered_with_gt_pose += 1
            
    n = len(pairs)
    print(f"Standard Search + Grid(3x3): Top-100 Recall = {recovered_with_estimated_pose}/{n} ({recovered_with_estimated_pose/n*100:.2f}%)")
    print(f"Dense Scale Search + Grid(3x3): Top-100 Recall = {recovered_with_gt_pose}/{n} ({recovered_with_gt_pose/n*100:.2f}%)")

if __name__ == "__main__":
    pairs = load_dev_data()
    evaluate_grid_extraction(pairs)
    evaluate_scale_search_enhancement(pairs)
