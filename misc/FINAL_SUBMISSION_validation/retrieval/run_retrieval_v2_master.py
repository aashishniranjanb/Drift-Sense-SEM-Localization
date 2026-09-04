"""
RETRIEVAL-V2 MASTER EXPERIMENT SUITE
=====================================
Target: Increase GT candidate recall from 105/140 (75.0%) -> >= 126/140 (90.0%+)
Preserves 100% of V54 baseline candidates in Ranks 1-200. Zero production changes.
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


# ─────────────────────────────────────────────────────────────
# FAILURE TAXONOMY MAPPING FOR 35 RETRIEVAL FAILURES
# ─────────────────────────────────────────────────────────────
FAILURE_TAXONOMY = {
    # NMS_SUPPRESSION (10 pairs): True GT suppressed by adjacent stronger replica (5-8px away)
    "pair_000": "NMS_SUPPRESSION", "pair_014": "NMS_SUPPRESSION", "pair_018": "NMS_SUPPRESSION",
    "pair_024": "NMS_SUPPRESSION", "pair_030": "NMS_SUPPRESSION", "pair_034": "NMS_SUPPRESSION",
    "pair_036": "NMS_SUPPRESSION", "pair_040": "NMS_SUPPRESSION", "pair_042": "NMS_SUPPRESSION",
    "pair_046": "NMS_SUPPRESSION",

    # LOW_SIGNAL (11 pairs): Weak correlation peak buried globally
    "pair_058": "LOW_SIGNAL", "pair_060": "LOW_SIGNAL", "pair_068": "LOW_SIGNAL",
    "pair_076": "LOW_SIGNAL", "pair_094": "LOW_SIGNAL", "pair_102": "LOW_SIGNAL",
    "pair_110": "LOW_SIGNAL", "pair_112": "LOW_SIGNAL", "pair_118": "LOW_SIGNAL",
    "pair_120": "LOW_SIGNAL", "pair_121": "LOW_SIGNAL",

    # SPATIAL / TRUNCATION (8 pairs): Boundary or peripheral targets
    "pair_002": "SPATIAL", "pair_056": "SPATIAL", "pair_075": "SPATIAL",
    "pair_085": "SPATIAL", "pair_091": "SPATIAL", "pair_104": "SPATIAL",
    "pair_119": "SPATIAL", "pair_124": "SPATIAL",

    # DEGRADATION (4 pairs): High noise or low contrast SEM
    "pair_031": "DEGRADATION", "pair_086": "DEGRADATION", "pair_108": "DEGRADATION",
    "pair_127": "DEGRADATION",

    # PERIODIC (2 pairs): Severe multi-pitch ambiguity
    "pair_134": "PERIODIC", "pair_138": "PERIODIC"
}


def subpixel_peak_refine(plane, px, py):
    H, W = plane.shape
    ix, iy = int(round(px)), int(round(py))
    if ix <= 0 or iy <= 0 or ix >= W - 1 or iy >= H - 1:
        return float(px), float(py)
    patch = plane[iy-1:iy+2, ix-1:ix+2]
    if patch.shape != (3, 3):
        return float(px), float(py)
    f00 = patch[1, 1]
    denom_x = 2 * (2 * f00 - patch[1, 2] - patch[1, 0])
    dx = (patch[1, 2] - patch[1, 0]) / denom_x if abs(denom_x) > 1e-5 else 0.0
    denom_y = 2 * (2 * f00 - patch[2, 1] - patch[0, 1])
    dy = (patch[2, 1] - patch[0, 1]) / denom_y if abs(denom_y) > 1e-5 else 0.0
    return float(ix + np.clip(dx, -0.5, 0.5)), float(iy + np.clip(dy, -0.5, 0.5))


def compute_gradient_magnitude(img):
    gx = cv2.Scharr(img.astype(np.float32), cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(img.astype(np.float32), cv2.CV_32F, 0, 1)
    return cv2.magnitude(gx, gy)


def extract_nms_peaks(plane, tw, th, radius=5, max_k=50, min_val=0.01):
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
        y1, y2 = max(0, py - radius), min(ch, py + radius + 1)
        x1, x2 = max(0, px - radius), min(cw, px + radius + 1)
        work[y1:y2, x1:x2] = -999.0
    return out


def extract_spatial_tile_peaks(plane, tw, th, grid_n=4, top_m=10):
    ch, cw = plane.shape[:2]
    tile_h = ch // grid_n
    tile_w = cw // grid_n
    out = []
    for gy in range(grid_n):
        for gx in range(grid_n):
            y1, y2 = gy * tile_h, (gy + 1) * tile_h if gy < grid_n - 1 else ch
            x1, x2 = gx * tile_w, (gx + 1) * tile_w if gx < grid_n - 1 else cw
            tile = plane[y1:y2, x1:x2]
            peaks = extract_nms_peaks(tile, tw, th, radius=3, max_k=top_m, min_val=0.01)
            for p in peaks:
                abs_px = p["peak_x"] + x1
                abs_py = p["peak_y"] + y1
                out.append({
                    "peak_x": abs_px, "peak_y": abs_py,
                    "cx": abs_px + tw / 2.0, "cy": abs_py + th / 2.0,
                    "score": p["score"]
                })
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
    if len(pv) < 4: return None

    abs_x = (lx + x0).astype(float)
    abs_y = (ly + y0).astype(float)
    order = np.argsort(pv)[::-1][:20]
    px_x, px_y = abs_x[order], abs_y[order]

    dists = np.hypot(px_x - ax, px_y - ay)
    keep = dists > 3.0
    px_x, px_y = px_x[keep], px_y[keep]
    if len(px_x) < 3: return None

    dx, dy = px_x - ax, px_y - ay
    h_vecs = [(x, y) for x, y in zip(dx, dy) if abs(y) < abs(x) * 0.4 and abs(x) > 3]
    v_vecs = [(x, y) for x, y in zip(dx, dy) if abs(x) < abs(y) * 0.4 and abs(y) > 3]

    if not h_vecs or not v_vecs:
        mags = np.hypot(dx, dy)
        idx6 = np.argsort(mags)[:6]
        h_vecs = [(dx[i], dy[i]) for i in idx6 if abs(dy[i]) <= abs(dx[i])]
        v_vecs = [(dx[i], dy[i]) for i in idx6 if abs(dx[i]) < abs(dy[i])]

    if not h_vecs or not v_vecs: return None

    vx_x = float(np.median([x for x, y in h_vecs]))
    vx_y = float(np.median([y for x, y in h_vecs]))
    vy_x = float(np.median([x for x, y in v_vecs]))
    vy_y = float(np.median([y for x, y in v_vecs]))

    return {"vx_x": vx_x, "vx_y": vx_y, "vy_x": vy_x, "vy_y": vy_y}


def process_pair_all_generators(args):
    pid, ref_p, srch_p, gt_x, gt_y, gt_found, est_scale, est_theta = args
    if gt_found == 0:
        return None

    ref = cv2.imread(ref_p, cv2.IMREAD_GRAYSCALE)
    srch = cv2.imread(srch_p, cv2.IMREAD_GRAYSCALE)
    if ref is None or srch is None:
        return None

    sh, sw = srch.shape[:2]
    tw = int(round(ref.shape[1] / est_scale))
    th = int(round(ref.shape[0] / est_scale))

    tpl = cv2.resize(ref.astype(np.float32), (tw, th), interpolation=cv2.INTER_AREA)
    tpl_rot = rotate_image(tpl, est_theta) if abs(est_theta) > 0.01 else tpl
    corr_intensity = cv2.matchTemplate(srch.astype(np.float32), tpl_rot, cv2.TM_CCOEFF_NORMED)

    # ── GENERATOR 0 (R0): V54 Baseline ──
    v25_cands = extract_candidates_akhilesh(corr_intensity, tw, th, ref, srch, est_scale, est_theta, max_final_k=200)

    # Save R0 Dump CSV
    dump_rows = []
    for rank, c in enumerate(v25_cands):
        dump_rows.append({
            "pair_id": pid, "candidate_rank": rank + 1,
            "x": float(c["cx"]), "y": float(c["cy"]),
            "ncc": float(c["corr_score"]),
            "GT_x": gt_x, "GT_y": gt_y,
            "err_to_gt": float(np.hypot(c["cx"] - gt_x, c["cy"] - gt_y))
        })
    os.makedirs("FINAL_SUBMISSION/validation/retrieval/v54_candidates", exist_ok=True)
    pd.DataFrame(dump_rows).to_csv(f"FINAL_SUBMISSION/validation/retrieval/v54_candidates/{pid}.csv", index=False)

    # Base union initialized with V54 candidates (1-200)
    union = []
    for c in v25_cands:
        sp_x, sp_y = subpixel_peak_refine(corr_intensity, c["peak_x"], c["peak_y"])
        union.append({
            "cx": float(sp_x + tw / 2.0), "cy": float(sp_y + th / 2.0),
            "peak_x": float(sp_x), "peak_y": float(sp_y),
            "score": float(c["corr_score"]), "source": "v54_baseline"
        })

    # Generator candidate pools dictionary
    pools = {"v54": list(union)}

    def append_to_union(cands, source_name, target_pool):
        for c in cands:
            cx, cy = c["cx"], c["cy"]
            dup = False
            for u in target_pool:
                if np.hypot(cx - u["cx"], cy - u["cy"]) < 2.0:
                    dup = True
                    if source_name not in u["source"]: u["source"] += f"+{source_name}"
                    break
            if not dup and len(target_pool) < 800:
                target_pool.append({
                    "cx": float(cx), "cy": float(cy),
                    "peak_x": float(c["peak_x"]), "peak_y": float(c["peak_y"]),
                    "score": float(c["score"]), "source": source_name
                })

    # ── GENERATOR 1 (R1): Multi-NMS (radii 2, 3, 5, 7, 10) ──
    nms_pool = []
    for r in [2, 3, 5, 7, 10]:
        peaks = extract_nms_peaks(corr_intensity, tw, th, radius=r, max_k=30)
        append_to_union(peaks, f"nms_r{r}", nms_pool)
    pools["multi_nms"] = nms_pool
    append_to_union(nms_pool, "multi_nms", union)

    # ── GENERATOR 2 (R2): Spatial Tile Harvesting (4x4 & 8x8) ──
    spatial_pool = []
    sp_4 = extract_spatial_tile_peaks(corr_intensity, tw, th, grid_n=4, top_m=5)
    sp_8 = extract_spatial_tile_peaks(corr_intensity, tw, th, grid_n=8, top_m=3)
    append_to_union(sp_4, "spatial_4x4", spatial_pool)
    append_to_union(sp_8, "spatial_8x8", spatial_pool)
    pools["spatial"] = spatial_pool
    append_to_union(spatial_pool, "spatial", union)

    # ── GENERATOR 3 (R3): Independent Gradient Retrieval ──
    ref_grad = compute_gradient_magnitude(tpl_rot)
    srch_grad = compute_gradient_magnitude(srch)
    corr_grad = cv2.matchTemplate(srch_grad, ref_grad, cv2.TM_CCOEFF_NORMED)
    grad_peaks = extract_nms_peaks(corr_grad, tw, th, radius=5, max_k=50)
    pools["gradient"] = grad_peaks
    append_to_union(grad_peaks, "gradient", union)

    # ── GENERATOR 4 (R4): Independent Phase Retrieval ──
    f_tpl = np.fft.fft2(tpl_rot, s=srch.shape)
    f_srch = np.fft.fft2(srch.astype(np.float32))
    eps = 1e-5
    cross_power = (f_srch * np.conj(f_tpl)) / (np.abs(f_srch * np.conj(f_tpl)) + eps)
    phase_plane = np.abs(np.fft.ifft2(cross_power))
    cp_h, cp_w = corr_intensity.shape
    phase_crop = phase_plane[:cp_h, :cp_w]
    if phase_crop.max() > 0: phase_crop /= phase_crop.max()
    phase_peaks = extract_nms_peaks(phase_crop, tw, th, radius=5, max_k=50)
    pools["phase"] = phase_peaks
    append_to_union(phase_peaks, "phase", union)

    # ── GENERATOR 5 (R5): Multi-Scale Context Retrieval ──
    ref_ds = cv2.resize(ref, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    srch_ds = cv2.resize(srch, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    tw_ds, th_ds = max(8, int(round(ref_ds.shape[1] / est_scale))), max(8, int(round(ref_ds.shape[0] / est_scale)))
    tpl_ds = cv2.resize(ref_ds.astype(np.float32), (tw_ds, th_ds), interpolation=cv2.INTER_AREA)
    tpl_ds_rot = rotate_image(tpl_ds, est_theta) if abs(est_theta) > 0.01 else tpl_ds
    corr_ctx = cv2.matchTemplate(srch_ds.astype(np.float32), tpl_ds_rot, cv2.TM_CCOEFF_NORMED)
    ctx_peaks_ds = extract_nms_peaks(corr_ctx, tw_ds, th_ds, radius=5, max_k=50)
    ctx_peaks = [{"cx": p["cx"]*2.0, "cy": p["cy"]*2.0, "peak_x": p["peak_x"]*2.0, "peak_y": p["peak_y"]*2.0, "score": p["score"]} for p in ctx_peaks_ds]
    pools["context"] = ctx_peaks
    append_to_union(ctx_peaks, "context", union)

    # ── GENERATOR 6 (R6): Local Lattice Probes (1st, 2nd, 3rd Order) ──
    lattice_pool = []
    if len(v25_cands) > 0:
        lat = estimate_local_pitch(corr_intensity, v25_cands[0]["peak_x"], v25_cands[0]["peak_y"])
        if lat is not None:
            vx_x, vx_y, vy_x, vy_y = lat["vx_x"], lat["vx_y"], lat["vy_x"], lat["vy_y"]
            for i in range(min(20, len(v25_cands))):
                cand_cx, cand_cy = v25_cands[i]["cx"], v25_cands[i]["cy"]
                for sx in [-3, -2, -1, 0, 1, 2, 3]:
                    for sy in [-3, -2, -1, 0, 1, 2, 3]:
                        if sx == 0 and sy == 0: continue
                        hx, hy = cand_cx + sx * vx_x + sy * vy_x, cand_cy + sx * vx_y + sy * vy_y
                        px_h, py_h = hx - tw / 2.0, hy - th / 2.0
                        if 0 <= px_h < cp_w and 0 <= py_h < cp_h:
                            sp_h_x, sp_h_y = subpixel_peak_refine(corr_intensity, px_h, py_h)
                            score_h = float(corr_intensity[int(round(py_h)), int(round(px_h))])
                            lattice_pool.append({
                                "cx": sp_h_x + tw / 2.0, "cy": sp_h_y + th / 2.0,
                                "peak_x": sp_h_x, "peak_y": sp_h_y, "score": score_h
                            })
    pools["lattice"] = lattice_pool
    append_to_union(lattice_pool, "lattice", union)

    # Evaluate min error for each generator and union
    def get_min_err(c_list, max_n=800):
        sub = c_list[:max_n]
        if not sub: return 999.0
        return min(np.hypot(c["cx"] - gt_x, c["cy"] - gt_y) for c in sub)

    err_union = get_min_err(union, 800)
    err_v54 = get_min_err(v25_cands, 200)

    # Failure category (if applicable)
    fail_cat = FAILURE_TAXONOMY.get(pid, "ACCEPTED" if err_v54 <= 5.0 else "UNCLASSIFIED")

    return {
        "pair_id": pid,
        "gt_x": gt_x, "gt_y": gt_y,
        "fail_category": fail_cat,
        "v54_hit": err_v54 <= 5.0,
        "union_hit_200": get_min_err(union, 200) <= 5.0,
        "union_hit_500": get_min_err(union, 500) <= 5.0,
        "union_hit_800": err_union <= 5.0,
        "union_min_err": float(err_union),
        "v54_min_err": float(err_v54),
        "v54_preserved": err_v54 <= 5.0 == (get_min_err(union[:200], 200) <= 5.0),
        "gen_hits": {g: get_min_err(p, 200) <= 5.0 for g, p in pools.items()},
        "union_candidates": len(union),
    }


def main():
    print("==================================================================")
    print("      RETRIEVAL-V2 MASTER EXPERIMENT RUNNER (7 GENERATORS)        ")
    print("==================================================================")
    print("Goal: Move GT Top200 recall from 105/140 (75.0%) -> >= 126/140 (90.0%+)\n")

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

    print(f"Running RETRIEVAL-V2 candidate extraction on {len(tasks)} pairs across 8 workers...")
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        for res in executor.map(process_pair_all_generators, tasks):
            if res is not None:
                results.append(res)
    elapsed = time.time() - t0
    print(f"Master extraction completed in {elapsed:.1f} seconds.\n")

    df_res = pd.DataFrame(results)
    n_present = len(df_res)

    # Baseline & Union Recall Summary
    v54_hits = int(df_res["v54_hit"].sum())
    union_200_hits = int(df_res["union_hit_200"].sum())
    union_500_hits = int(df_res["union_hit_500"].sum())
    union_800_hits = int(df_res["union_hit_800"].sum())

    print("=" * 65)
    print("          RETRIEVAL-V2 MASTER CANDIDATE RECALL SUMMARY")
    print("=" * 65)
    print(f" Baseline V54 (Top 200):   {v54_hits:3d} / {n_present}  ({v54_hits/n_present*100:5.1f}%)")
    print(f" Union Top 200:            {union_200_hits:3d} / {n_present}  ({union_200_hits/n_present*100:5.1f}%)")
    print(f" Union Top 500:            {union_500_hits:3d} / {n_present}  ({union_500_hits/n_present*100:5.1f}%)")
    print(f" Union Top 800 (Full):     {union_800_hits:3d} / {n_present}  ({union_800_hits/n_present*100:5.1f}%)  <-- TARGET: >= 126")
    print("-" * 65)

    # Breakdown by Failure Class
    print("\nRECOVERY BREAKDOWN BY RETRIEVAL FAILURE CLASS (35 PAIRS):")
    print(f"{'Category':<20s} | {'Total':<6s} | {'Baseline V54':<14s} | {'Union Recovered':<16s} | {'Recovery Rate':<15s}")
    print("-" * 75)

    failure_summary = {}
    for cat in ["NMS_SUPPRESSION", "LOW_SIGNAL", "SPATIAL", "DEGRADATION", "PERIODIC"]:
        sub = df_res[df_res["fail_category"] == cat]
        tot = len(sub)
        base_h = int(sub["v54_hit"].sum())
        union_h = int(sub["union_hit_800"].sum())
        rec_pct = (union_h / tot * 100.0) if tot > 0 else 0.0
        failure_summary[cat] = {"total": tot, "baseline": base_h, "recovered": union_h, "rate_pct": round(rec_pct, 1)}
        print(f"{cat:<20s} | {tot:<6d} | {base_h:<14d} | {union_h:<16d} | {rec_pct:5.1f}%")

    print("=" * 65)

    # Save Machine-Readable JSON & CSV
    report_json = {
        "benchmark_pairs": n_present,
        "baseline_v54_top200_hits": v54_hits,
        "union_top200_hits": union_200_hits,
        "union_top500_hits": union_500_hits,
        "union_top800_hits": union_800_hits,
        "union_recall_pct": round(union_800_hits / n_present * 100.0, 2),
        "retrieval_failures_recovered": union_800_hits - v54_hits,
        "target_achieved": (union_800_hits >= 126),
        "v54_anchors_preserved": int(df_res["v54_preserved"].sum()),
        "failure_class_breakdown": failure_summary
    }

    with open("FINAL_SUBMISSION/validation/retrieval/retrieval_v2_report.json", "w") as f:
        json.dump(report_json, f, indent=2)

    df_res.to_csv("FINAL_SUBMISSION/validation/retrieval/retrieval_v2_candidates.csv", index=False)
    print("\nSaved JSON: FINAL_SUBMISSION/validation/retrieval/retrieval_v2_report.json")
    print("Saved CSV:  FINAL_SUBMISSION/validation/retrieval/retrieval_v2_candidates.csv")


if __name__ == "__main__":
    main()
