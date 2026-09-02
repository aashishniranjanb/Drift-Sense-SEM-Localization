import os
import sys
import cv2
import numpy as np
import pandas as pd

sys.path.append("phase2")
from scale_search import coarse_to_fine_scale_search
from rotation_search import coarse_to_fine_rotation_search

def inspect_missed_cases():
    csv_path = "data/phase2_dev/pairs.csv"
    df = pd.read_csv(csv_path)
    present_df = df[df["gt_found"] == 1].copy()
    
    records = []
    
    for _, row in present_df.iterrows():
        ref_path = os.path.join("data/phase2_dev", row["reference_path"])
        search_path = os.path.join("data/phase2_dev", row["search_path"])
        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
        
        gt_x = float(row["gt_x"])
        gt_y = float(row["gt_y"])
        gt_scale = float(row.get("gt_scale", 10.0)) if "gt_scale" in row else 10.0
        gt_rotation = float(row.get("gt_rotation", 0.0)) if "gt_rotation" in row else 0.0
        
        scale_res = coarse_to_fine_scale_search(ref_img, search_img)
        rot_res = coarse_to_fine_rotation_search(scale_res["best_template"], search_img)
        
        corr_plane = rot_res["corr_plane"]
        rot_tpl = rot_res["rotated_template"]
        th, tw = rot_tpl.shape[:2]
        ch, cw = corr_plane.shape[:2]
        
        # NMS r=5 extraction
        work = corr_plane.copy()
        candidates = []
        for _ in range(100):
            _, max_val, _, max_loc = cv2.minMaxLoc(work)
            if max_val <= 0.01 or np.isnan(max_val):
                break
            px, py = max_loc
            cx = px + tw / 2.0
            cy = py + th / 2.0
            candidates.append({"px": px, "py": py, "cx": cx, "cy": cy, "score": max_val})
            y1, y2 = max(0, py - 5), min(ch, py + 6)
            x1, x2 = max(0, px - 5), min(cw, px + 6)
            work[y1:y2, x1:x2] = -999.0
            
        retrieved_top100 = False
        retrieved_rank = None
        for rank, c in enumerate(candidates):
            if np.hypot(c["cx"] - gt_x, c["cy"] - gt_y) <= 5.0:
                retrieved_top100 = True
                retrieved_rank = rank
                break
                
        # Check GT position in corr plane
        gt_px = int(round(gt_x - tw / 2.0))
        gt_py = int(round(gt_y - th / 2.0))
        
        in_bounds = (0 <= gt_px < cw and 0 <= gt_py < ch)
        gt_corr_val = float(corr_plane[gt_py, gt_px]) if in_bounds else -1.0
        max_corr_val = float(np.max(corr_plane))
        
        # Calculate raw rank of the GT pixel in the entire correlation plane
        raw_rank = None
        if in_bounds:
            raw_rank = int(np.sum(corr_plane > gt_corr_val))
            
        records.append({
            "pair_id": row["pair_id"],
            "set_type": row["set_type"],
            "est_scale": scale_res["best_scale"],
            "gt_scale": gt_scale,
            "scale_error": abs(scale_res["best_scale"] - gt_scale),
            "est_rot": rot_res["best_theta"],
            "gt_rot": gt_rotation,
            "rot_error": abs(rot_res["best_theta"] - gt_rotation),
            "gt_x": gt_x,
            "gt_y": gt_y,
            "retrieved_top100": int(retrieved_top100),
            "retrieved_rank": retrieved_rank if retrieved_rank is not None else -1,
            "gt_corr_val": gt_corr_val,
            "max_corr_val": max_corr_val,
            "raw_rank": raw_rank if raw_rank is not None else -1
        })
        
    df_res = pd.DataFrame(records)
    df_res.to_csv("results/phase2/V12_MAIN_TRACK/V12_DEEP_MISS_INSPECTION.csv", index=False)
    
    missed = df_res[df_res["retrieved_top100"] == 0]
    print(f"Total evaluated: {len(df_res)} | Missed in Top-100: {len(missed)} ({len(missed)/len(df_res)*100:.1f}%)")
    print(f"Missed breakdown by Set:")
    print(missed["set_type"].value_counts())
    
    print("\nScale error statistics for Missed vs Retrieved:")
    print("Retrieved mean scale error:", df_res[df_res["retrieved_top100"] == 1]["scale_error"].mean())
    print("Missed mean scale error:   ", missed["scale_error"].mean())
    
    print("\nRotation error statistics for Missed vs Retrieved:")
    print("Retrieved mean rot error:  ", df_res[df_res["retrieved_top100"] == 1]["rot_error"].mean())
    print("Missed mean rot error:     ", missed["rot_error"].mean())
    
    print("\nTop 15 worst missed cases by raw_rank:")
    print(missed.sort_values(by="raw_rank", ascending=False)[["pair_id", "set_type", "scale_error", "rot_error", "gt_corr_val", "max_corr_val", "raw_rank"]].head(15))

if __name__ == "__main__":
    inspect_missed_cases()
