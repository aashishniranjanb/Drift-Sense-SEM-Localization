import os
import cv2
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path("c:/Users/Home/Downloads/PROJECTS/Drift-Sense-SEM-Localization")
sys.path.append(str(PROJECT_ROOT))

from fallbacks.pose_fallback import perform_pose_fallback_search
import importlib.util
spec = importlib.util.spec_from_file_location("candidate_extractor", str(PROJECT_ROOT / "team/akhilesh-localization/candidate_extractor.py"))
candidate_extractor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate_extractor)
extract_nms_fast = candidate_extractor.extract_nms_fast

def r0_extractor(corr_plane, tw, th, search_img, est_scale, max_final_k=50, pool_size=200):
    sh, sw = search_img.shape[:2] if search_img is not None else (1024, 1024)
    search_cx, search_cy = sw / 2.0, sh / 2.0
    
    pool = extract_nms_fast(corr_plane, tw, th, max_k=pool_size, r=5)
    if len(pool) <= max_final_k:
        return pool
        
    center_queue = []
    periphery_queue = []
    
    for c in pool:
        d = np.hypot(c["cx"] - search_cx, c["cy"] - search_cy)
        c["dist_to_center"] = float(d)
        if d <= 260.0:
            center_queue.append(c)
        else:
            periphery_queue.append(c)
            
    for c in center_queue:
        c["center_priority"] = c["corr_score"] - 0.05 * (c["dist_to_center"] / 260.0) ** 2
    center_queue.sort(key=lambda x: x["center_priority"], reverse=True)
    periphery_queue.sort(key=lambda x: x["corr_score"], reverse=True)
    
    n_center = min(35 if max_final_k == 50 else int(max_final_k * 0.7), len(center_queue))
    n_periph = min(max_final_k - n_center, len(periphery_queue))
    
    final_cands = center_queue[:n_center] + periphery_queue[:n_periph]
    
    if len(final_cands) < max_final_k:
        remaining = center_queue[n_center:] + periphery_queue[n_periph:]
        remaining.sort(key=lambda x: x["corr_score"], reverse=True)
        final_cands.extend(remaining[:max_final_k - len(final_cands)])
        
    return final_cands[:max_final_k]

def r1_extractor(corr_plane, tw, th, search_img, est_scale):
    return r0_extractor(corr_plane, tw, th, search_img, est_scale, max_final_k=100, pool_size=500)

def find_periodic_spacing(cands, num_top=20):
    if len(cands) < 3:
        return None
    pts = np.array([[c['cx'], c['cy']] for c in cands[:num_top]])
    
    # Compute pairwise distances
    from scipy.spatial.distance import pdist, squareform
    dists = pdist(pts)
    if len(dists) == 0:
        return None
        
    # Get histogram of distances to find dominant spacing
    hist, bin_edges = np.histogram(dists, bins=30, range=(10, 200))
    peak_bin = np.argmax(hist)
    dominant_dist = (bin_edges[peak_bin] + bin_edges[peak_bin+1])/2.0
    return dominant_dist

def r2_extractor(corr_plane, tw, th, search_img, est_scale):
    cands = r1_extractor(corr_plane, tw, th, search_img, est_scale)
    
    # Add periodic rescue
    spacing = find_periodic_spacing(cands)
    if spacing is not None and spacing > 20:
        # Simple rescue: extrapolate from top candidates
        rescued = []
        c_set = {(int(c['cx']/10), int(c['cy']/10)) for c in cands}
        
        for c in cands[:10]:
            for dx in [-spacing, 0, spacing]:
                for dy in [-spacing, 0, spacing]:
                    if dx == 0 and dy == 0:
                        continue
                    nx = c['cx'] + dx
                    ny = c['cy'] + dy
                    
                    if nx < tw/2 or nx > search_img.shape[1] - tw/2 or ny < th/2 or ny > search_img.shape[0] - th/2:
                        continue
                        
                    grid_key = (int(nx/10), int(ny/10))
                    if grid_key not in c_set:
                        # Grab corr score
                        px, py = int(nx - tw/2), int(ny - th/2)
                        if 0 <= py < corr_plane.shape[0] and 0 <= px < corr_plane.shape[1]:
                            score = float(corr_plane[py, px])
                            if score > 0.3: # Threshold for rescue
                                rescued.append({
                                    "peak_x": px, "peak_y": py,
                                    "cx": nx, "cy": ny,
                                    "corr_score": score,
                                    "is_rescued": True
                                })
                                c_set.add(grid_key)
        
        rescued.sort(key=lambda x: x['corr_score'], reverse=True)
        cands.extend(rescued)
        cands.sort(key=lambda x: x['corr_score'], reverse=True)
        
    return cands[:200]

def r3_extractor(corr_plane, tw, th, search_img, est_scale):
    # R2 + adaptive quota weighting
    # We simulate this by adjusting weights based on spacing confidence
    pool = extract_nms_fast(corr_plane, tw, th, max_k=500, r=5)
    
    sh, sw = search_img.shape[:2] if search_img is not None else (1024, 1024)
    search_cx, search_cy = sw / 2.0, sh / 2.0
    
    spacing = find_periodic_spacing(pool)
    
    center_weight = 0.05
    if spacing is not None and spacing > 20:
        center_weight = 0.02 # Less center bias if highly periodic
        
    center_queue = []
    periphery_queue = []
    for c in pool:
        d = np.hypot(c["cx"] - search_cx, c["cy"] - search_cy)
        c["dist_to_center"] = float(d)
        if d <= 260.0:
            c["center_priority"] = c["corr_score"] - center_weight * (c["dist_to_center"] / 260.0) ** 2
            center_queue.append(c)
        else:
            periphery_queue.append(c)
            
    center_queue.sort(key=lambda x: x.get("center_priority", 0), reverse=True)
    periphery_queue.sort(key=lambda x: x["corr_score"], reverse=True)
    
    max_final_k = 100
    n_center = min(int(max_final_k * 0.7), len(center_queue))
    n_periph = min(max_final_k - n_center, len(periphery_queue))
    
    cands = center_queue[:n_center] + periphery_queue[:n_periph]
    if len(cands) < max_final_k:
        remaining = center_queue[n_center:] + periphery_queue[n_periph:]
        remaining.sort(key=lambda x: x["corr_score"], reverse=True)
        cands.extend(remaining[:max_final_k - len(cands)])
        
    # Same rescue as R2
    if spacing is not None and spacing > 20:
        rescued = []
        c_set = {(int(c['cx']/10), int(c['cy']/10)) for c in cands}
        
        for c in cands[:10]:
            for dx in [-spacing, 0, spacing]:
                for dy in [-spacing, 0, spacing]:
                    if dx == 0 and dy == 0:
                        continue
                    nx = c['cx'] + dx
                    ny = c['cy'] + dy
                    if nx < tw/2 or nx > sw - tw/2 or ny < th/2 or ny > sh - th/2:
                        continue
                        
                    grid_key = (int(nx/10), int(ny/10))
                    if grid_key not in c_set:
                        px, py = int(nx - tw/2), int(ny - th/2)
                        if 0 <= py < corr_plane.shape[0] and 0 <= px < corr_plane.shape[1]:
                            score = float(corr_plane[py, px])
                            if score > 0.3:
                                rescued.append({
                                    "peak_x": px, "peak_y": py,
                                    "cx": nx, "cy": ny,
                                    "corr_score": score
                                })
                                c_set.add(grid_key)
        rescued.sort(key=lambda x: x['corr_score'], reverse=True)
        cands.extend(rescued)
        
    cands.sort(key=lambda x: x['corr_score'], reverse=True)
    return cands[:200]

def main():
    pairs_csv = PROJECT_ROOT / "data/phase2_dev/pairs.csv"
    df = pd.read_csv(pairs_csv)
    
    df_present = df[df["gt_found"] == 1]
    print(f"Total PRESENT pairs to evaluate: {len(df_present)}")
    
    results = []
    
    for _, row in tqdm(df_present.iterrows(), total=len(df_present)):
        ref_path = PROJECT_ROOT / f'data/phase2_dev/{row["reference_path"]}'
        search_path = PROJECT_ROOT / f'data/phase2_dev/{row["search_path"]}'
        
        ref_img = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(str(search_path), cv2.IMREAD_GRAYSCALE)
        
        if ref_img is None or search_img is None:
            continue
            
        gt_x, gt_y = row["gt_x"], row["gt_y"]
        
        # We need est_scale to determine tw, th
        est_scale = 10.0 # Standard scale in Drift-Sense
        tw = int(round(ref_img.shape[1] / est_scale))
        th = int(round(ref_img.shape[0] / est_scale))
        
        # tolerance
        tol_x = max(25, tw * 0.25)
        tol_y = max(25, th * 0.25)
        
        # 1. Pose estimation
        t0 = time.time()
        result = perform_pose_fallback_search(ref_img, search_img)
        corr_plane = result['corr_plane']
        t_pose = time.time() - t0
        
        def run_variant(variant_fn, name):
            t1 = time.time()
            cands = variant_fn(corr_plane, tw, th, search_img, est_scale)
            t_ext = time.time() - t1
            
            # Find rank of GT
            gt_rank = -1
            for i, c in enumerate(cands):
                if abs(c["cx"] - gt_x) <= tol_x and abs(c["cy"] - gt_y) <= tol_y:
                    gt_rank = i + 1
                    break
                    
            return {
                "pair_id": row["pair_id"],
                "variant": name,
                "gt_rank": gt_rank,
                "latency_sec": t_ext
            }
            
        results.append(run_variant(r0_extractor, "R0"))
        results.append(run_variant(r1_extractor, "R1"))
        results.append(run_variant(r2_extractor, "R2"))
        results.append(run_variant(r3_extractor, "R3"))
        
    res_df = pd.DataFrame(results)
    
    out_dir = PROJECT_ROOT / "phase2/V22_CHAMPIONSHIP/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_dir / "blast1_retrieval_results.csv", index=False)
    
    # Compute recall
    def recall_at_k(variant_df, k):
        return (variant_df["gt_rank"].apply(lambda x: 1 if 0 < x <= k else 0).mean()) * 100
        
    summary = []
    for v in ["R0", "R1", "R2", "R3"]:
        v_df = res_df[res_df["variant"] == v]
        summary.append({
            "Variant": v,
            "Top-10 (%)": recall_at_k(v_df, 10),
            "Top-20 (%)": recall_at_k(v_df, 20),
            "Top-50 (%)": recall_at_k(v_df, 50),
            "Top-100 (%)": recall_at_k(v_df, 100),
            "Top-200 (%)": recall_at_k(v_df, 200),
            "Latency (ms)": v_df["latency_sec"].mean() * 1000
        })
        
    sum_df = pd.DataFrame(summary)
    print(sum_df)
    
    # Write markdown report
    md_lines = ["# Blast 1 Retrieval Audit Report", "", "## Results", ""]
    md_lines.append(sum_df.to_markdown(index=False))
    md_lines.append("")
    md_lines.append("## Conclusion")
    
    top50_r3 = float(sum_df[sum_df["Variant"] == "R3"]["Top-50 (%)"].iloc[0])
    top50_r0 = float(sum_df[sum_df["Variant"] == "R0"]["Top-50 (%)"].iloc[0])
    delta = top50_r3 - top50_r0
    
    if top50_r3 >= 78.0:
        md_lines.append(f"**DECISION: KEEP**. R3 achieved {top50_r3:.2f}% Top-50 recall (Δ = +{delta:.2f}% vs R0), exceeding the 78% threshold.")
    else:
        md_lines.append(f"**DECISION: REJECT**. R3 achieved {top50_r3:.2f}% Top-50 recall (Δ = +{delta:.2f}% vs R0), which is below the 78% threshold.")
        
    with open(out_dir / "BLAST1_REPORT.md", "w") as f:
        f.write("\n".join(md_lines))
        
if __name__ == "__main__":
    main()
