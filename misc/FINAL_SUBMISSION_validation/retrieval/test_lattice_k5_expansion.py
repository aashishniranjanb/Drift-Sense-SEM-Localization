"""
TEST 4TH AND 5TH ORDER LATTICE PROBE EXPANSION (k=4,5)
======================================================
Expands local pitch lattice search grid from k=1..3 to k=1..5 to test recovery of moderate drift cases.
"""

import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "FINAL_SUBMISSION/runtime/src")
sys.path.insert(0, "FINAL_SUBMISSION/validation/retrieval")
from utils import rotate_image
from candidate_extractor import extract_candidates_akhilesh
from build_retrieval_v2 import subpixel_peak_refine, estimate_local_pitch, extract_peaks_from_plane, compute_gradient_magnitude

def extract_multi_source_union_k5(ref_img, search_img, est_scale, est_theta, max_total_k=800):
    sh, sw = search_img.shape[:2]
    tw = int(round(ref_img.shape[1] / est_scale))
    th = int(round(ref_img.shape[0] / est_scale))

    if tw <= 0 or th <= 0 or tw >= sw or th >= sh:
        return []

    tpl = cv2.resize(ref_img.astype(np.float32), (tw, th), interpolation=cv2.INTER_AREA)
    tpl_rot = rotate_image(tpl, est_theta) if abs(est_theta) > 0.01 else tpl

    # 1. Primary V25 Baseline Pool
    corr_intensity = cv2.matchTemplate(search_img.astype(np.float32), tpl_rot, cv2.TM_CCOEFF_NORMED)
    v25_cands = extract_candidates_akhilesh(corr_intensity, tw, th, ref_img, search_img, est_scale, est_theta, max_final_k=200)

    union = []
    for c in v25_cands:
        sp_x, sp_y = subpixel_peak_refine(corr_intensity, c["peak_x"], c["peak_y"])
        union.append({
            "cx": float(sp_x + tw / 2.0), "cy": float(sp_y + th / 2.0),
            "peak_x": float(sp_x), "peak_y": float(sp_y),
            "score": float(c["corr_score"]), "source": "v25_intensity"
        })

    def append_unique(candidates, source_name):
        for c in candidates:
            cx, cy = c["cx"], c["cy"]
            dup = False
            for u in union:
                if np.hypot(cx - u["cx"], cy - u["cy"]) < 2.0:
                    dup = True
                    if source_name not in u["source"]:
                        u["source"] += f"+{source_name}"
                    break
            if not dup and len(union) < max_total_k:
                union.append({
                    "cx": float(cx), "cy": float(cy),
                    "peak_x": float(c["peak_x"]), "peak_y": float(c["peak_y"]),
                    "score": float(c["score"]), "source": source_name
                })

    # 2. Gradient
    ref_grad = compute_gradient_magnitude(tpl_rot)
    srch_grad = compute_gradient_magnitude(search_img)
    corr_grad = cv2.matchTemplate(srch_grad, ref_grad, cv2.TM_CCOEFF_NORMED)
    grad_peaks = extract_peaks_from_plane(corr_grad, tw, th, max_k=50)
    append_unique(grad_peaks, "gradient")

    # 3. Phase Correlation
    f_tpl = np.fft.fft2(tpl_rot, s=search_img.shape)
    f_srch = np.fft.fft2(search_img.astype(np.float32))
    eps = 1e-5
    cross_power = (f_srch * np.conj(f_tpl)) / (np.abs(f_srch * np.conj(f_tpl)) + eps)
    phase_plane = np.abs(np.fft.ifft2(cross_power))
    cp_h, cp_w = corr_intensity.shape
    phase_crop = phase_plane[:cp_h, :cp_w]
    p_max = phase_crop.max()
    if p_max > 0: phase_crop = phase_crop / p_max
    phase_peaks = extract_peaks_from_plane(phase_crop, tw, th, max_k=50)
    append_unique(phase_peaks, "phase")

    # 4. Multi-scale Context
    ref_ds = cv2.resize(ref_img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    srch_ds = cv2.resize(search_img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    tw_ds = max(8, int(round(ref_ds.shape[1] / est_scale)))
    th_ds = max(8, int(round(ref_ds.shape[0] / est_scale)))
    tpl_ds = cv2.resize(ref_ds.astype(np.float32), (tw_ds, th_ds), interpolation=cv2.INTER_AREA)
    tpl_ds_rot = rotate_image(tpl_ds, est_theta) if abs(est_theta) > 0.01 else tpl_ds
    corr_ctx = cv2.matchTemplate(srch_ds.astype(np.float32), tpl_ds_rot, cv2.TM_CCOEFF_NORMED)
    ctx_peaks_ds = extract_peaks_from_plane(corr_ctx, tw_ds, th_ds, max_k=50)
    ctx_peaks = [{"cx": cp["cx"] * 2.0, "cy": cp["cy"] * 2.0, "peak_x": cp["peak_x"] * 2.0, "peak_y": cp["peak_y"] * 2.0, "score": cp["score"]} for cp in ctx_peaks_ds]
    append_unique(ctx_peaks, "context_multi_scale")

    # 5. EXPANDED 1st - 5th Order Local Lattice Probes (k=1..5)
    if len(v25_cands) > 0:
        top_px, top_py = v25_cands[0]["peak_x"], v25_cands[0]["peak_y"]
        lat = estimate_local_pitch(corr_intensity, top_px, top_py)
        if lat is not None:
            vx_x, vx_y = lat["vx_x"], lat["vx_y"]
            vy_x, vy_y = lat["vy_x"], lat["vy_y"]
            lattice_probes = []
            for i in range(min(20, len(v25_cands))):
                cand_cx, cand_cy = v25_cands[i]["cx"], v25_cands[i]["cy"]
                for sx in [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]:
                    for sy in [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]:
                        if sx == 0 and sy == 0: continue
                        hx = cand_cx + sx * vx_x + sy * vy_x
                        hy = cand_cy + sx * vx_y + sy * vy_y
                        px_h = hx - tw / 2.0
                        py_h = hy - th / 2.0
                        if 0 <= px_h < cp_w - 1 and 0 <= py_h < cp_h - 1:
                            sp_h_x, sp_h_y = subpixel_peak_refine(corr_intensity, px_h, py_h)
                            score_h = float(corr_intensity[int(round(py_h)), int(round(px_h))])
                            lattice_probes.append({
                                "cx": sp_h_x + tw / 2.0, "cy": sp_h_y + th / 2.0,
                                "peak_x": sp_h_x, "peak_y": sp_h_y,
                                "score": score_h
                            })
            append_unique(lattice_probes, "local_lattice_subpixel_k5")

    return union

def process_pair_k5(args):
    pid, ref_p, srch_p, gt_x, gt_y, gt_found, est_scale, est_theta = args
    if gt_found == 0:
        return None

    ref = cv2.imread(ref_p, cv2.IMREAD_GRAYSCALE)
    srch = cv2.imread(srch_p, cv2.IMREAD_GRAYSCALE)
    if ref is None or srch is None:
        return None

    union = extract_multi_source_union_k5(ref, srch, est_scale, est_theta, max_total_k=800)
    errs = [np.hypot(c["cx"] - gt_x, c["cy"] - gt_y) for c in union]
    min_err = min(errs) if errs else 999.0

    return {
        "pair_id": pid,
        "gt_found": 1,
        "total_cands": len(union),
        "hit_5px": min_err <= 5.0,
        "hit_10px": min_err <= 10.0,
        "min_err": min_err
    }

def main():
    print("=" * 65)
    print("  TEST 4TH & 5TH ORDER LATTICE PROBE EXPANSION (k=1..5)")
    print("=" * 65)

    pairs_df = pd.read_csv("data/phase2_dev/pairs.csv")
    v25_df = pd.read_csv("data/phase2_dev/v25_predictions.csv")

    tasks = []
    for _, row in pairs_df.iterrows():
        pid = row["pair_id"]
        v25_r = v25_df[v25_df["pair_id"] == pid].iloc[0]
        ref_p = os.path.join("data/phase2_dev", row["reference_path"].replace("\\", "/"))
        srch_p = os.path.join("data/phase2_dev", row["search_path"].replace("\\", "/"))
        tasks.append((
            pid, ref_p, srch_p,
            float(row.get("gt_x", 0.0)), float(row.get("gt_y", 0.0)),
            int(row["gt_found"]),
            float(v25_r["scale"]), float(v25_r["theta"])
        ))

    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        for res in executor.map(process_pair_k5, tasks):
            if res is not None:
                results.append(res)

    df_res = pd.DataFrame(results)
    hits_5 = df_res["hit_5px"].sum()
    hits_10 = df_res["hit_10px"].sum()
    n = len(df_res)

    print("\n" + "=" * 65)
    print(f"  k=1..5 LATTICE PROBE RECALL: {hits_5} / {n} ({hits_5/n*100.0:.1f}%) <= 5px")
    print(f"  k=1..5 LATTICE PROBE RECALL: {hits_10} / {n} ({hits_10/n*100.0:.1f}%) <= 10px")
    print("=" * 65)

if __name__ == "__main__":
    main()
