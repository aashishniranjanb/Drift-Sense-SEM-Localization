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
            "set_type": row["set_type"]
        })
    return pairs

def extract_local_maxima(corr_plane, w=3, min_score=0.01):
    size = 2 * w + 1
    local_max = (maximum_filter(corr_plane, size=size) == corr_plane) & (corr_plane > min_score)
    y_idx, x_idx = np.where(local_max)
    scores = corr_plane[y_idx, x_idx]
    sorted_order = np.argsort(scores)[::-1]
    return [{"px": int(x_idx[i]), "py": int(y_idx[i]), "score": float(scores[i]), "source": "local_max"} for i in sorted_order]

def extract_percentile_peaks(corr_plane, percentile=99.0):
    thresh = np.percentile(corr_plane, percentile)
    if thresh <= 0.05:
        thresh = 0.05
    y_idx, x_idx = np.where(corr_plane >= thresh)
    scores = corr_plane[y_idx, x_idx]
    sorted_order = np.argsort(scores)[::-1]
    return [{"px": int(x_idx[i]), "py": int(y_idx[i]), "score": float(scores[i]), "source": "threshold"} for i in sorted_order]

def extract_nms_peaks(corr_plane, r=10, max_k=100):
    work = corr_plane.copy()
    ch, cw = work.shape
    peaks = []
    for _ in range(max_k):
        _, max_val, _, max_loc = cv2.minMaxLoc(work)
        if max_val <= -1.0 or np.isnan(max_val):
            break
        px, py = max_loc
        peaks.append({"px": int(px), "py": int(py), "score": float(max_val), "source": "nms"})
        y1, y2 = max(0, py - r), min(ch, py + r + 1)
        x1, x2 = max(0, px - r), min(cw, px + r + 1)
        work[y1:y2, x1:x2] = -999.0
    return peaks

def deduplicate_candidates(candidates, tw, th, dedup_radius=3.0, max_k=100):
    candidates = sorted(candidates, key=lambda c: c["score"], reverse=True)
    unique = []
    for c in candidates:
        cx = c["px"] + tw / 2.0
        cy = c["py"] + th / 2.0
        is_dup = False
        for u in unique:
            if np.hypot(cx - u["cx"], cy - u["cy"]) < dedup_radius:
                is_dup = True
                break
        if not is_dup:
            unique.append({
                "peak_x": c["px"],
                "peak_y": c["py"],
                "cx": cx,
                "cy": cy,
                "score": c["score"],
                "source": c.get("source", "unknown")
            })
            if len(unique) >= max_k:
                break
    return unique

def evaluate_retrieval(candidates, gt_x, gt_y, thresholds=(20, 50, 100), loc_tol=5.0):
    results = {k: 0 for k in thresholds}
    best_rank = None
    min_dist = float("inf")
    for rank, c in enumerate(candidates):
        dist = np.hypot(c["cx"] - gt_x, c["cy"] - gt_y)
        if dist < min_dist:
            min_dist = dist
        if dist <= loc_tol and best_rank is None:
            best_rank = rank
            
    if best_rank is not None:
        for k in thresholds:
            if best_rank < k:
                results[k] = 1
    return results, best_rank, min_dist

# Pre-compute coarse correlation planes for single-hypothesis to allow ultra-fast sweeps
def precompute_nominal_corr_planes(pairs):
    print("Pre-computing nominal scale/rotation correlation planes for all 140 pairs...")
    cached = []
    for p in pairs:
        ref_img = p["ref_img"]
        search_img = p["search_img"]
        
        scale_res = coarse_to_fine_scale_search(ref_img, search_img)
        rot_res = coarse_to_fine_rotation_search(scale_res["best_template"], search_img)
        
        corr_plane = rot_res["corr_plane"]
        rot_tpl = rot_res["rotated_template"]
        th, tw = rot_tpl.shape[:2]
        
        # Gradient corr plane
        search_grad = extract_gradient(search_img)
        rot_tpl_grad = extract_gradient(rot_tpl)
        corr_grad = cv2.matchTemplate(search_grad, rot_tpl_grad, cv2.TM_CCOEFF_NORMED)
        
        cached.append({
            "pair_id": p["pair_id"],
            "corr_plane": corr_plane,
            "corr_grad": corr_grad,
            "tw": tw,
            "th": th,
            "gt_x": p["gt_x"],
            "gt_y": p["gt_y"],
            "best_scale": scale_res["best_scale"],
            "best_theta": rot_res["best_theta"],
            "set_type": p["set_type"]
        })
    print("Pre-computation complete.")
    return cached

# 1. Sweep NMS radius vs Local Max window
def run_nms_window_grid(cached_data):
    print("\n--- [V12.2] Running NMS Radius & Local Max Window Grid Sweep ---")
    radii = [1, 2, 3, 4, 5, 7, 10, 12, 15]
    windows = [1, 2, 3, 4, 5, 7]
    
    rows = []
    
    # 1. Pure NMS sweep
    for r in radii:
        t0 = time.perf_counter()
        tot_hits = {20: 0, 50: 0, 100: 0}
        for item in cached_data:
            cands = extract_nms_peaks(item["corr_plane"], r=r, max_k=100)
            dedup = deduplicate_candidates(cands, item["tw"], item["th"], dedup_radius=float(r), max_k=100)
            hits, _, _ = evaluate_retrieval(dedup, item["gt_x"], item["gt_y"])
            for k in [20, 50, 100]:
                tot_hits[k] += hits[k]
        latency = (time.perf_counter() - t0) / len(cached_data) * 1000
        n = len(cached_data)
        rows.append({
            "method": f"NMS(r={r})",
            "param": r,
            "top20": tot_hits[20] / n * 100,
            "top50": tot_hits[50] / n * 100,
            "top100": tot_hits[100] / n * 100,
            "latency_ms": latency
        })
        print(f"NMS r={r:2d} -> Top-20: {tot_hits[20]/n*100:.1f}%, Top-50: {tot_hits[50]/n*100:.1f}%, Top-100: {tot_hits[100]/n*100:.1f}% ({latency:.1f} ms)")
        
    # 2. Pure Local Max sweep
    for w in windows:
        t0 = time.perf_counter()
        tot_hits = {20: 0, 50: 0, 100: 0}
        for item in cached_data:
            cands = extract_local_maxima(item["corr_plane"], w=w)
            dedup = deduplicate_candidates(cands, item["tw"], item["th"], dedup_radius=float(w), max_k=100)
            hits, _, _ = evaluate_retrieval(dedup, item["gt_x"], item["gt_y"])
            for k in [20, 50, 100]:
                tot_hits[k] += hits[k]
        latency = (time.perf_counter() - t0) / len(cached_data) * 1000
        n = len(cached_data)
        rows.append({
            "method": f"LocalMax(w={w})",
            "param": w,
            "top20": tot_hits[20] / n * 100,
            "top50": tot_hits[50] / n * 100,
            "top100": tot_hits[100] / n * 100,
            "latency_ms": latency
        })
        print(f"LocalMax w={w:2d} -> Top-20: {tot_hits[20]/n*100:.1f}%, Top-50: {tot_hits[50]/n*100:.1f}%, Top-100: {tot_hits[100]/n*100:.1f}% ({latency:.1f} ms)")
        
    df_res = pd.DataFrame(rows)
    df_res.to_csv("results/phase2/V12_MAIN_TRACK/V12_NMS_WINDOW_SWEEP.csv", index=False)
    return df_res

# 2. Threshold / Percentile Peak Sweep (V12.3)
def run_percentile_sweep(cached_data):
    print("\n--- [V12.3] Running Thresholded/Percentile Correlation Sweep ---")
    percentiles = [99.9, 99.5, 99.0, 98.0, 97.0, 95.0]
    rows = []
    
    for p in percentiles:
        tot_hits = {20: 0, 50: 0, 100: 0}
        for item in cached_data:
            c_lmax = extract_local_maxima(item["corr_plane"], w=3)
            c_pct = extract_percentile_peaks(item["corr_plane"], percentile=p)
            c_union = c_lmax + c_pct
            dedup = deduplicate_candidates(c_union, item["tw"], item["th"], dedup_radius=3.0, max_k=100)
            hits, _, _ = evaluate_retrieval(dedup, item["gt_x"], item["gt_y"])
            for k in [20, 50, 100]:
                tot_hits[k] += hits[k]
        n = len(cached_data)
        rows.append({
            "percentile": p,
            "top20": tot_hits[20] / n * 100,
            "top50": tot_hits[50] / n * 100,
            "top100": tot_hits[100] / n * 100
        })
        print(f"Union(LMax w=3 + Pct={p:4.1f}%) -> Top-20: {tot_hits[20]/n*100:.1f}%, Top-50: {tot_hits[50]/n*100:.1f}%, Top-100: {tot_hits[100]/n*100:.1f}%")
        
    df_res = pd.DataFrame(rows)
    df_res.to_csv("results/phase2/V12_MAIN_TRACK/V12_PERCENTILE_SWEEP.csv", index=False)
    return df_res

# 3. Multi-Hypothesis Pose Sweep (V12.4)
def run_pose_hypothesis_sweep(pairs):
    print("\n--- [V12.4] Running Multi-Hypothesis Pose Sweep (H=1..7) ---")
    h_values = [1, 2, 3, 4, 5, 7]
    rows = []
    
    scale_min, scale_max = 8.0, 12.0
    coarse_scales = np.arange(scale_min, scale_max + 1e-5, 0.5)
    
    for H in h_values:
        t0 = time.perf_counter()
        tot_hits = {20: 0, 50: 0, 100: 0}
        
        for p in pairs:
            ref_f = p["ref_img"].astype(np.float32)
            search_f = p["search_img"].astype(np.float32)
            ref_h, ref_w = ref_f.shape[:2]
            
            # Coarse scale search
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
            
            for s_hyp, _ in top_scales:
                # Refine scale
                f_scales = np.arange(max(scale_min, s_hyp - 0.3), min(scale_max, s_hyp + 0.31), 0.1)
                best_f_score = -1.0
                best_f_scale = s_hyp
                best_f_tpl = None
                for fs in f_scales:
                    tw = int(round(ref_w / fs))
                    th = int(round(ref_h / fs))
                    tpl = cv2.resize(ref_f, (tw, th), interpolation=cv2.INTER_AREA)
                    res = cv2.matchTemplate(search_f, tpl, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(res)
                    if max_val > best_f_score:
                        best_f_score = max_val
                        best_f_scale = fs
                        best_f_tpl = tpl
                        
                rot_res = coarse_to_fine_rotation_search(best_f_tpl, p["search_img"])
                corr = rot_res["corr_plane"]
                rot_tpl = rot_res["rotated_template"]
                th, tw = rot_tpl.shape[:2]
                final_tw, final_th = tw, th
                
                # Extract candidates from this hypothesis using local max (w=3) + percentile
                c_lm = extract_local_maxima(corr, w=3)
                c_pct = extract_percentile_peaks(corr, percentile=99.0)
                for c in c_lm + c_pct:
                    union_candidates.append(c)
                    
            dedup = deduplicate_candidates(union_candidates, final_tw, final_th, dedup_radius=3.0, max_k=100)
            hits, _, _ = evaluate_retrieval(dedup, p["gt_x"], p["gt_y"])
            for k in [20, 50, 100]:
                tot_hits[k] += hits[k]
                
        latency = (time.perf_counter() - t0) / len(pairs) * 1000
        n = len(pairs)
        rows.append({
            "hypotheses": H,
            "top20": tot_hits[20] / n * 100,
            "top50": tot_hits[50] / n * 100,
            "top100": tot_hits[100] / n * 100,
            "latency_ms": latency
        })
        print(f"Multi-Hypothesis H={H} -> Top-20: {tot_hits[20]/n*100:.1f}%, Top-50: {tot_hits[50]/n*100:.1f}%, Top-100: {tot_hits[100]/n*100:.1f}% (Latency: {latency:.1f} ms)")
        
    df_res = pd.DataFrame(rows)
    df_res.to_csv("results/phase2/V12_MAIN_TRACK/V12_POSE_HYPOTHESIS_SWEEP.csv", index=False)
    return df_res

# 4. Multi-Channel Retrieval (V12.5) & Rescue Taxonomy (V12.6)
def run_channel_union_and_rescue(pairs):
    print("\n--- [V12.5 & V12.6] Multi-Channel Candidate Union & Rescue Analysis ---")
    
    # We will test Multi-Channel with H=3 pose hypotheses
    scale_min, scale_max = 8.0, 12.0
    coarse_scales = np.arange(scale_min, scale_max + 1e-5, 0.5)
    
    rescue_records = []
    tot_hits = {20: 0, 50: 0, 100: 0}
    
    for p in pairs:
        ref_f = p["ref_img"].astype(np.float32)
        search_f = p["search_img"].astype(np.float32)
        ref_h, ref_w = ref_f.shape[:2]
        
        search_grad = extract_gradient(p["search_img"])
        
        # Coarse scale search on intensity
        coarse_scores = []
        for s in coarse_scales:
            tw = int(round(ref_w / s))
            th = int(round(ref_h / s))
            tpl = cv2.resize(ref_f, (tw, th), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(search_f, tpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            coarse_scores.append((s, max_val))
        coarse_scores.sort(key=lambda x: x[1], reverse=True)
        top_scales = coarse_scores[:3]  # H=3
        
        union_cands = []
        final_tw, final_th = None, None
        
        for s_hyp, _ in top_scales:
            f_scales = np.arange(max(scale_min, s_hyp - 0.3), min(scale_max, s_hyp + 0.31), 0.1)
            best_f_score = -1.0
            best_f_scale = s_hyp
            best_f_tpl = None
            for fs in f_scales:
                tw = int(round(ref_w / fs))
                th = int(round(ref_h / fs))
                tpl = cv2.resize(ref_f, (tw, th), interpolation=cv2.INTER_AREA)
                res = cv2.matchTemplate(search_f, tpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                if max_val > best_f_score:
                    best_f_score = max_val
                    best_f_scale = fs
                    best_f_tpl = tpl
                    
            rot_res = coarse_to_fine_rotation_search(best_f_tpl, p["search_img"])
            corr_int = rot_res["corr_plane"]
            rot_tpl = rot_res["rotated_template"]
            th, tw = rot_tpl.shape[:2]
            final_tw, final_th = tw, th
            
            # Gradient correlation
            rot_tpl_grad = extract_gradient(rot_tpl)
            corr_grad = cv2.matchTemplate(search_grad, rot_tpl_grad, cv2.TM_CCOEFF_NORMED)
            
            # Extract from Intensity (LocalMax w=2, w=4 + Threshold 99.0 + NMS r=5)
            c_int_lm2 = extract_local_maxima(corr_int, w=2)
            c_int_lm4 = extract_local_maxima(corr_int, w=4)
            c_int_pct = extract_percentile_peaks(corr_int, percentile=99.0)
            c_int_nms = extract_nms_peaks(corr_int, r=5, max_k=50)
            
            # Extract from Gradient
            c_grad_lm3 = extract_local_maxima(corr_grad, w=3)
            c_grad_pct = extract_percentile_peaks(corr_grad, percentile=99.0)
            
            for c in c_int_lm2 + c_int_lm4 + c_int_pct + c_int_nms + c_grad_lm3 + c_grad_pct:
                union_cands.append(c)
                
        dedup = deduplicate_candidates(union_cands, final_tw, final_th, dedup_radius=3.0, max_k=100)
        hits, best_rank, min_dist = evaluate_retrieval(dedup, p["gt_x"], p["gt_y"])
        for k in [20, 50, 100]:
            tot_hits[k] += hits[k]
            
        # Determine failure cause for any pair failing Top-100
        rescue_cat = "RETRIEVED_TOP100"
        if hits[100] == 0:
            # Check if GT is in intensity corr plane at nominal
            gt_px = int(round(p["gt_x"] - final_tw / 2.0))
            gt_py = int(round(p["gt_y"] - final_th / 2.0))
            ch, cw = corr_int.shape[:2]
            if not (0 <= gt_px < cw and 0 <= gt_py < ch):
                rescue_cat = "RESCUE_OUT_OF_BOUNDS"
            else:
                gt_val = corr_int[gt_py, gt_px]
                if gt_val < 0.10:
                    rescue_cat = "RESCUE_SCALE_ROT_MISMATCH"
                else:
                    rescue_cat = "RESCUE_DENSITY_CAP"
                    
        rescue_records.append({
            "pair_id": p["pair_id"],
            "set_type": p["set_type"],
            "gt_x": p["gt_x"],
            "gt_y": p["gt_y"],
            "best_rank": best_rank if best_rank is not None else -1,
            "min_dist": min_dist,
            "top20": hits[20],
            "top50": hits[50],
            "top100": hits[100],
            "rescue_category": rescue_cat
        })
        
    n = len(pairs)
    print(f"\n=======================================================")
    print(f"   V12 MULTI-CHANNEL + MULTI-HYPOTHESIS UNION RESULTS  ")
    print(f"=======================================================")
    print(f"Top-20 Recall:  {tot_hits[20]}/{n} ({tot_hits[20]/n*100:.2f}%)")
    print(f"Top-50 Recall:  {tot_hits[50]}/{n} ({tot_hits[50]/n*100:.2f}%)")
    print(f"Top-100 Recall: {tot_hits[100]}/{n} ({tot_hits[100]/n*100:.2f}%)")
    print(f"=======================================================")
    
    df_rescue = pd.DataFrame(rescue_records)
    df_rescue.to_csv("results/phase2/V12_MAIN_TRACK/V12_RESCUE_AUDIT.csv", index=False)
    
    tax_counts = df_rescue["rescue_category"].value_counts().to_dict()
    print("\nRescue Category Breakdown:")
    for cat, count in tax_counts.items():
        print(f" - {cat}: {count} ({count/n*100:.1f}%)")
        
    with open("results/phase2/V12_MAIN_TRACK/V12_RESCUE_AUDIT.md", "w") as f:
        f.write(f"""# V12 Candidate Rescue Audit

## Multi-Channel + Multi-Pose Candidate Recovery Performance
- **Top-20 Recall**: {tot_hits[20]/n*100:.2f}% ({tot_hits[20]}/{n})
- **Top-50 Recall**: {tot_hits[50]/n*100:.2f}% ({tot_hits[50]}/{n})
- **Top-100 Recall**: {tot_hits[100]/n*100:.2f}% ({tot_hits[100]}/{n})

## Failure Cause Breakdown for Missed Candidates
""")
        for cat, count in tax_counts.items():
            f.write(f"- **{cat}**: {count} ({count/n*100:.2f}%)\n")
            
    return df_rescue

def main():
    pairs = load_dev_data()
    cached = precompute_nominal_corr_planes(pairs)
    
    # 1. NMS & Window sweep
    run_nms_window_grid(cached)
    
    # 2. Percentile sweep
    run_percentile_sweep(cached)
    
    # 3. Multi-Hypothesis Pose sweep
    run_pose_hypothesis_sweep(pairs)
    
    # 4. Multi-Channel Union & Rescue taxonomy
    run_channel_union_and_rescue(pairs)

if __name__ == "__main__":
    main()
