"""
RETRIEVAL-V2 WITH SUBPIXEL PARABOLIC PEAK INTERPOLATION
======================================================
Applies 2D 3x3 parabolic subpixel peak interpolation to all candidate correlation peaks
to refine candidate peak coordinates (px + dx, py + dy).
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import maximum_filter
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "FINAL_SUBMISSION/runtime/src")
from utils import rotate_image
from candidate_extractor import extract_candidates_akhilesh


def subpixel_peak_refine(plane, px, py):
    """Applies 3x3 2D quadratic subpixel peak interpolation around (px, py)."""
    H, W = plane.shape
    ix, iy = int(round(px)), int(round(py))
    if ix <= 0 or iy <= 0 or ix >= W - 1 or iy >= H - 1:
        return float(px), float(py)

    patch = plane[iy-1:iy+2, ix-1:ix+2]
    if patch.shape != (3, 3):
        return float(px), float(py)

    # Quadratic fit along x: dx = (f(1,0) - f(-1,0)) / (2 * (2*f(0,0) - f(1,0) - f(-1,0)))
    f00 = patch[1, 1]
    f10 = patch[1, 2]; f_10 = patch[1, 0]
    denom_x = 2 * (2 * f00 - f10 - f_10)
    dx = (f10 - f_10) / denom_x if abs(denom_x) > 1e-5 else 0.0

    # Quadratic fit along y: dy = (f(0,1) - f(0,-1)) / (2 * (2*f(0,0) - f(0,1) - f(0,-1)))
    f01 = patch[2, 1]; f0_1 = patch[0, 1]
    denom_y = 2 * (2 * f00 - f01 - f0_1)
    dy = (f01 - f0_1) / denom_y if abs(denom_y) > 1e-5 else 0.0

    dx = np.clip(dx, -0.5, 0.5)
    dy = np.clip(dy, -0.5, 0.5)
    return float(ix + dx), float(iy + dy)


def compute_gradient_magnitude(img):
    gx = cv2.Scharr(img.astype(np.float32), cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(img.astype(np.float32), cv2.CV_32F, 0, 1)
    return cv2.magnitude(gx, gy)


def extract_peaks_from_plane(plane, tw, th, max_k=100, min_val=0.01, min_dist=5):
    ch, cw = plane.shape[:2]
    work = plane.copy()
    out = []
    for rank in range(max_k):
        _, mv, _, ml = cv2.minMaxLoc(work)
        if mv <= min_val or np.isnan(mv):
            break
        px, py = ml
        sp_x, sp_y = subpixel_peak_refine(plane, px, py)
        out.append({"peak_x": sp_x, "peak_y": sp_y, "cx": sp_x + tw / 2.0, "cy": sp_y + th / 2.0,
                    "score": float(mv), "raw_rank": rank + 1})
        y1, y2 = max(0, py - min_dist), min(ch, py + min_dist + 1)
        x1, x2 = max(0, px - min_dist), min(cw, px + min_dist + 1)
        work[y1:y2, x1:x2] = -999.0
    return out


def estimate_local_pitch(corr_plane, anchor_px, anchor_py, search_radius=120):
    H, W = corr_plane.shape
    ax, ay = int(round(anchor_px)), int(round(anchor_py))
    x0, x1 = max(0, ax - search_radius), min(W, ax + search_radius)
    y0, y1 = max(0, ay - search_radius), min(H, ay + search_radius)
    patch = corr_plane[y0:y1, x0:x1]

    lmax = maximum_filter(patch, size=7)
    mask = (patch == lmax) & (patch > 0.15)
    ly, lx = np.where(mask)
    pv = patch[ly, lx]
    if len(pv) < 4:
        return None

    abs_x = (lx + x0).astype(float)
    abs_y = (ly + y0).astype(float)
    order = np.argsort(pv)[::-1][:20]
    px_x, px_y = abs_x[order], abs_y[order]

    dists = np.hypot(px_x - ax, px_y - ay)
    keep = dists > 3.0
    px_x, px_y = px_x[keep], px_y[keep]
    if len(px_x) < 3:
        return None

    dx, dy = px_x - ax, px_y - ay
    h_vecs = [(x, y) for x, y in zip(dx, dy) if abs(y) < abs(x) * 0.4 and abs(x) > 3]
    v_vecs = [(x, y) for x, y in zip(dx, dy) if abs(x) < abs(y) * 0.4 and abs(y) > 3]

    if not h_vecs or not v_vecs:
        mags = np.hypot(dx, dy)
        idx6 = np.argsort(mags)[:6]
        h_vecs = [(dx[i], dy[i]) for i in idx6 if abs(dy[i]) <= abs(dx[i])]
        v_vecs = [(dx[i], dy[i]) for i in idx6 if abs(dx[i]) < abs(dy[i])]

    if not h_vecs or not v_vecs:
        return None

    vx_x = float(np.median([x for x, y in h_vecs]))
    vx_y = float(np.median([y for x, y in h_vecs]))
    vy_x = float(np.median([x for x, y in v_vecs]))
    vy_y = float(np.median([y for x, y in v_vecs]))

    return {"vx_x": vx_x, "vx_y": vx_y, "vy_x": vy_x, "vy_y": vy_y}


def extract_multi_source_union(ref_img, search_img, est_scale, est_theta, max_total_k=800):
    sh, sw = search_img.shape[:2]
    tw = int(round(ref_img.shape[1] / est_scale))
    th = int(round(ref_img.shape[0] / est_scale))

    if tw <= 0 or th <= 0 or tw >= sw or th >= sh:
        return []

    tpl = cv2.resize(ref_img.astype(np.float32), (tw, th), interpolation=cv2.INTER_AREA)
    tpl_rot = rotate_image(tpl, est_theta) if abs(est_theta) > 0.01 else tpl

    # 1. Primary V25 Baseline Pool (with subpixel refinement)
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

    # 2. Source B: Gradient Correlation
    ref_grad = compute_gradient_magnitude(tpl_rot)
    srch_grad = compute_gradient_magnitude(search_img)
    corr_grad = cv2.matchTemplate(srch_grad, ref_grad, cv2.TM_CCOEFF_NORMED)
    grad_peaks = extract_peaks_from_plane(corr_grad, tw, th, max_k=50)
    append_unique(grad_peaks, "gradient")

    # 3. Source C: Phase Correlation
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

    # 4. Source D: Multi-scale Context Correlation
    ref_ds = cv2.resize(ref_img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    srch_ds = cv2.resize(search_img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    tw_ds = max(8, int(round(ref_ds.shape[1] / est_scale)))
    th_ds = max(8, int(round(ref_ds.shape[0] / est_scale)))
    tpl_ds = cv2.resize(ref_ds.astype(np.float32), (tw_ds, th_ds), interpolation=cv2.INTER_AREA)
    tpl_ds_rot = rotate_image(tpl_ds, est_theta) if abs(est_theta) > 0.01 else tpl_ds
    corr_ctx = cv2.matchTemplate(srch_ds.astype(np.float32), tpl_ds_rot, cv2.TM_CCOEFF_NORMED)
    ctx_peaks_ds = extract_peaks_from_plane(corr_ctx, tw_ds, th_ds, max_k=50)

    ctx_peaks = []
    for cp in ctx_peaks_ds:
        ctx_peaks.append({
            "cx": cp["cx"] * 2.0, "cy": cp["cy"] * 2.0,
            "peak_x": cp["peak_x"] * 2.0, "peak_y": cp["peak_y"] * 2.0,
            "score": cp["score"]
        })
    append_unique(ctx_peaks, "context_multi_scale")

    # 5. Source E: 1st, 2nd, and 3rd Order Local Lattice Probes
    if len(v25_cands) > 0:
        top_px, top_py = v25_cands[0]["peak_x"], v25_cands[0]["peak_y"]
        lat = estimate_local_pitch(corr_intensity, top_px, top_py)
        if lat is not None:
            vx_x, vx_y = lat["vx_x"], lat["vx_y"]
            vy_x, vy_y = lat["vy_x"], lat["vy_y"]
            lattice_probes = []
            for i in range(min(20, len(v25_cands))):
                cand_cx, cand_cy = v25_cands[i]["cx"], v25_cands[i]["cy"]
                for sx in [-3, -2, -1, 0, 1, 2, 3]:
                    for sy in [-3, -2, -1, 0, 1, 2, 3]:
                        if sx == 0 and sy == 0: continue
                        hx = cand_cx + sx * vx_x + sy * vy_x
                        hy = cand_cy + sx * vx_y + sy * vy_y
                        px_h = hx - tw / 2.0
                        py_h = hy - th / 2.0
                        if 0 <= px_h < cp_w and 0 <= py_h < cp_h:
                            sp_h_x, sp_h_y = subpixel_peak_refine(corr_intensity, px_h, py_h)
                            score_h = float(corr_intensity[int(round(py_h)), int(round(px_h))])
                            lattice_probes.append({
                                "cx": sp_h_x + tw / 2.0, "cy": sp_h_y + th / 2.0,
                                "peak_x": sp_h_x, "peak_y": sp_h_y,
                                "score": score_h
                            })
            append_unique(lattice_probes, "local_lattice_subpixel")

    return union


def process_pair(args):
    pid, ref_p, srch_p, gt_x, gt_y, gt_found, est_scale, est_theta = args
    if gt_found == 0:
        return {"pair_id": pid, "gt_found": 0}

    ref = cv2.imread(ref_p, cv2.IMREAD_GRAYSCALE)
    srch = cv2.imread(srch_p, cv2.IMREAD_GRAYSCALE)
    if ref is None or srch is None:
        return None

    union = extract_multi_source_union(ref, srch, est_scale, est_theta, max_total_k=800)
    errs = [np.hypot(c["cx"] - gt_x, c["cy"] - gt_y) for c in union]

    def min_e(k):
        sub = errs[:k]
        return min(sub) if len(sub) > 0 else 999.0

    return {
        "pair_id": pid,
        "gt_found": 1,
        "gt_x": gt_x, "gt_y": gt_y,
        "total_cands": len(union),
        "hit_1": min_e(1) <= 5.0,
        "hit_5": min_e(5) <= 5.0,
        "hit_10": min_e(10) <= 5.0,
        "hit_20": min_e(20) <= 5.0,
        "hit_50": min_e(50) <= 5.0,
        "hit_100": min_e(100) <= 5.0,
        "hit_200": min_e(200) <= 5.0,
        "hit_300": min_e(300) <= 5.0,
        "hit_500": min_e(500) <= 5.0,
        "hit_800": min_e(800) <= 5.0,
        "min_err_200": float(min_e(200)),
        "min_err_800": float(min_e(800)),
    }


def main():
    print("=" * 65)
    print("  RETRIEVAL-V2: SUBPIXEL PARABOLIC PEAK REFINEMENT UNION")
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

    print(f"Extracting multi-source candidate unions across {len(tasks)} pairs (8 workers)...")
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        for res in executor.map(process_pair, tasks):
            if res is not None:
                results.append(res)
    print(f"Done in {time.time()-t0:.1f}s. Processed {len(results)} pairs.\n")

    df_res = pd.DataFrame([r for r in results if r["gt_found"] == 1])
    n = len(df_res)

    print("=" * 65)
    print("      RETRIEVAL-V2 TOP-K CANDIDATE RECALL (140 PRESENT PAIRS)")
    print("=" * 65)
    for k in [1, 5, 10, 20, 50, 100, 200, 300, 500, 800]:
        col = f"hit_{k}"
        hits = df_res[col].sum()
        pct = hits / n * 100.0
        tag = "  <-- V25 BASELINE ANCHOR RECALL" if k == 200 else ""
        tag = "  <-- SUBPIXEL MULTI-SOURCE + LATTICE UNION RECALL" if k == 800 else tag
        print(f"  Top {k:<3d}: {hits:3d} / {n}  ({pct:5.1f}%){tag}")
    print("=" * 65)

    os.makedirs("FINAL_SUBMISSION/validation/retrieval", exist_ok=True)
    df_res.to_csv("FINAL_SUBMISSION/validation/retrieval/retrieval_v2_pool_audit.csv", index=False)
    print("Saved audit CSV to FINAL_SUBMISSION/validation/retrieval/retrieval_v2_pool_audit.csv")


if __name__ == "__main__":
    main()
