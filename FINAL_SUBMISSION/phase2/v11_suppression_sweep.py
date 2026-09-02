import pandas as pd
import numpy as np
import cv2
import sys
import os
import time

sys.path.append("phase2")
from scale_search import coarse_to_fine_scale_search
from rotation_search import coarse_to_fine_rotation_search

def run_suppression_sweep():
    df = pd.read_csv("data/phase2_dev/pairs.csv")
    present_df = df[df["gt_found"] == 1]
    total_present = len(present_df)
    
    sweep_radii = [1, 2, 3, 5, 8, 10, 15, 20, 30]
    results = []
    
    # Pre-load image pairs to avoid slow disk I/O in loop
    print("Pre-loading images...")
    pairs_data = []
    for idx, r in present_df.iterrows():
        ref_img = cv2.imread("data/phase2_dev/" + r["reference_path"], cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread("data/phase2_dev/" + r["search_path"], cv2.IMREAD_GRAYSCALE)
        pairs_data.append((r["pair_id"], ref_img, search_img, r["gt_x"], r["gt_y"]))
        
    print("Running suppression radius sweep...")
    for r_val in sweep_radii:
        hits = {10: 0, 20: 0, 50: 0, 100: 0}
        
        t0 = time.perf_counter()
        
        for pair_id, ref_img, search_img, gt_x, gt_y in pairs_data:
            scale_res = coarse_to_fine_scale_search(ref_img, search_img)
            est_scale = scale_res["best_scale"]
            best_template = scale_res["best_template"]
            
            rot_res = coarse_to_fine_rotation_search(best_template, search_img)
            corr = rot_res["corr_plane"]
            th, tw = rot_res["rotated_template"].shape[:2]
            ch, cw = corr.shape[:2]
            
            # Peak suppression with radius r_val
            work = corr.copy()
            best_rank = None
            for rank in range(100):
                _, max_val, _, max_loc = cv2.minMaxLoc(work)
                if max_val <= -1.0 or np.isnan(max_val): break
                px, py = max_loc
                cx, cy = px + tw / 2.0, py + th / 2.0
                
                # Check if this candidate matches GT
                if np.hypot(cx - gt_x, cy - gt_y) <= 5.0:
                    best_rank = rank
                    break
                    
                # Suppress neighborhood
                y1, y2 = max(0, py - r_val), min(ch, py + r_val + 1)
                x1, x2 = max(0, px - r_val), min(cw, px + r_val + 1)
                work[y1:y2, x1:x2] = -999.0
                
            if best_rank is not None:
                if best_rank < 10: hits[10] += 1
                if best_rank < 20: hits[20] += 1
                if best_rank < 50: hits[50] += 1
                if best_rank < 100: hits[100] += 1
                
        t1 = time.perf_counter()
        avg_latency_ms = ((t1 - t0) / total_present) * 1000.0
        
        rec_10 = (hits[10] / total_present) * 100
        rec_20 = (hits[20] / total_present) * 100
        rec_50 = (hits[50] / total_present) * 100
        rec_100 = (hits[100] / total_present) * 100
        
        print(f"Radius={r_val:2d} | Top-10={rec_10:.1f}%, Top-20={rec_20:.1f}%, Top-50={rec_50:.1f}%, Top-100={rec_100:.1f}% | Latency={avg_latency_ms:.1f}ms")
        
        results.append({
            "radius": r_val,
            "top10_recall": rec_10,
            "top20_recall": rec_20,
            "top50_recall": rec_50,
            "top100_recall": rec_100,
            "latency_ms": avg_latency_ms
        })
        
    # Write CSV
    os.makedirs("results/phase2/V11_MAIN_TRACK", exist_ok=True)
    sweep_df = pd.DataFrame(results)
    sweep_df.to_csv("results/phase2/V11_MAIN_TRACK/suppression_sweep.csv", index=False)
    
    # Write Markdown
    md_lines = [
        "# V11 Candidate Peak Suppression Radius Sweep\n",
        "| Radius (px) | Top-10 Recall (%) | Top-20 Recall (%) | Top-50 Recall (%) | Top-100 Recall (%) | Latency (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |"
    ]
    for r in results:
        md_lines.append(f"| {r['radius']} | {r['top10_recall']:.2f}% | {r['top20_recall']:.2f}% | {r['top50_recall']:.2f}% | {r['top100_recall']:.2f}% | {r['latency_ms']:.1f}ms |")
        
    with open("results/phase2/V11_MAIN_TRACK/suppression_sweep.md", "w") as f:
        f.write("\n".join(md_lines))
    print("Report written to results/phase2/V11_MAIN_TRACK/suppression_sweep.md")

if __name__ == "__main__":
    run_suppression_sweep()
