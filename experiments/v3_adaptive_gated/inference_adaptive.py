"""
Drift-Sense++ Adaptive Structural Registration Engine
Confidence-Gated Multi-Stage Architecture:
- Stage 1: Fast Global Retrieval (Intensity + Scharr Gradient)
- Stage 2: Confidence Assessment (Peak Gap Delta-S, PSR, Peak Score)
- Stage 3: Adaptive Execution Cascade (Fast Path <40ms, Normal Path ~100ms, Hard Path ~350ms)
- Stage 4: Multi-Feature Verification & Structural Consistency Metric C_i = mu / (sigma + eps)
- Stage 5: Conditional Periodic Center Tie-Breaker (Invoked ONLY on true structural ties)
- Stage 6: Subpixel 2D Paraboloid Refinement
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import cv2

from anchor_consensus import select_distinctive_anchors


def normalize_intensity(image: np.ndarray) -> np.ndarray:
    """Normalizes image intensity to float32 [0, 1] with percentile scaling."""
    img_f = image.astype(np.float32)
    p_low, p_high = np.percentile(img_f, (1, 99))
    if p_high > p_low:
        img_norm = np.clip((img_f - p_low) / (p_high - p_low), 0.0, 1.0)
    else:
        img_norm = img_f / 255.0
    return img_norm.astype(np.float32)


def extract_gradient_and_orientation(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extracts Scharr gradient magnitude [0, 1] and orientation angle [-pi, pi]."""
    img_f = image.astype(np.float32) / 255.0 if image.dtype == np.uint8 else image.astype(np.float32)
    gx = cv2.Scharr(img_f, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(img_f, cv2.CV_32F, 0, 1)
    g_mag = cv2.magnitude(gx, gy)
    g_ang = np.arctan2(gy, gx)
    max_val = g_mag.max()
    if max_val > 1e-6:
        g_mag /= max_val
    return g_mag.astype(np.float32), g_ang.astype(np.float32)


def compute_psr(corr_plane: np.ndarray, peak_x: int, peak_y: int, sidelobe_radius: int = 6) -> float:
    """Computes Peak-to-Sidelobe Ratio (PSR) of correlation plane."""
    h, w = corr_plane.shape
    mask = np.ones((h, w), dtype=bool)

    y_min = max(0, peak_y - sidelobe_radius)
    y_max = min(h, peak_y + sidelobe_radius + 1)
    x_min = max(0, peak_x - sidelobe_radius)
    x_max = min(w, peak_x + sidelobe_radius + 1)

    mask[y_min:y_max, x_min:x_max] = False
    sidelobe = corr_plane[mask]
    if len(sidelobe) == 0 or np.std(sidelobe) < 1e-7:
        return 0.0

    peak_val = corr_plane[peak_y, peak_x]
    mean_side = float(np.mean(sidelobe))
    std_side = float(np.std(sidelobe))

    psr = (peak_val - mean_side) / std_side
    return float(psr)


def extract_distinct_spatial_peaks(corr_map: np.ndarray, top_k: int = 6, min_dist: int = 15) -> list[dict]:
    """Extracts distinct spatial peaks using Non-Maximum Suppression."""
    work = corr_map.copy()
    h, w = work.shape
    peaks = []
    for _ in range(top_k):
        _, max_val, _, max_loc = cv2.minMaxLoc(work)
        if max_val <= -1.0 or np.isnan(max_val):
            break
        px, py = max_loc
        peaks.append({"x": int(px), "y": int(py), "score": float(max_val)})
        y1, y2 = max(0, py - min_dist), min(h, py + min_dist + 1)
        x1, x2 = max(0, px - min_dist), min(w, px + min_dist + 1)
        work[y1:y2, x1:x2] = -999.0
    return peaks


def local_phase_correlation(ref_patch: np.ndarray, search_patch: np.ndarray) -> tuple[float, float, float]:
    """Computes local phase correlation between reference and candidate search patch."""
    h, w = ref_patch.shape
    sh, sw = search_patch.shape
    if (h, w) != (sh, sw):
        ref_patch = cv2.resize(ref_patch, (sw, sh), interpolation=cv2.INTER_AREA)

    ref_f = ref_patch.astype(np.float32)
    search_f = search_patch.astype(np.float32)

    hann = cv2.createHanningWindow((sw, sh), cv2.CV_32F)
    ref_win = (ref_f - np.mean(ref_f)) * hann
    search_win = (search_f - np.mean(search_f)) * hann

    G_a = np.fft.fft2(ref_win)
    G_b = np.fft.fft2(search_win)
    cross_power = G_a * np.conj(G_b)
    magnitude = np.abs(cross_power) + 1e-7
    r = np.real(np.fft.ifft2(cross_power / magnitude))
    r_shift = np.fft.fftshift(r)

    cy, cx = sh // 2, sw // 2
    win_r = 5
    sub_r = r_shift[max(0, cy - win_r):min(sh, cy + win_r + 1),
                    max(0, cx - win_r):min(sw, cx + win_r + 1)]

    _, max_val, _, max_loc = cv2.minMaxLoc(sub_r)
    dx = float(max_loc[0] - win_r)
    dy = float(max_loc[1] - win_r)
    phase_score = float(max_val)

    return dx, dy, phase_score


def subpixel_refine_2d(corr_plane: np.ndarray, int_x: int, int_y: int) -> tuple[float, float]:
    """Fits 2D quadratic paraboloid surface on 5x5 neighborhood around correlation peak."""
    h, w = corr_plane.shape
    if int_x < 2 or int_x >= w - 2 or int_y < 2 or int_y >= h - 2:
        return float(int_x), float(int_y)

    patch = corr_plane[int_y-2:int_y+3, int_x-2:int_x+3].astype(np.float64)
    y_coords, x_coords = np.mgrid[-2:3, -2:3]

    X_mat = np.column_stack([
        x_coords.ravel()**2,
        y_coords.ravel()**2,
        x_coords.ravel() * y_coords.ravel(),
        x_coords.ravel(),
        y_coords.ravel(),
        np.ones(25)
    ])
    Z_vec = patch.ravel()

    try:
        coeff, _, _, _ = np.linalg.lstsq(X_mat, Z_vec, rcond=None)
        a, b, c, d, e, _ = coeff
        denom = 4 * a * b - c**2
        if abs(denom) > 1e-6 and a < 0 and b < 0:
            dx = (c * e - 2 * b * d) / denom
            dy = (c * d - 2 * a * e) / denom
            dx = np.clip(dx, -1.5, 1.5)
            dy = np.clip(dy, -1.5, 1.5)
            return float(int_x + dx), float(int_y + dy)
    except Exception:
        pass

    return float(int_x), float(int_y)


def perform_adaptive_localization(
    ref_img: np.ndarray,
    search_img: np.ndarray,
    verbose: bool = False
) -> tuple[float, float, dict]:
    """
    Main Drift-Sense++ Adaptive Structural Registration Pipeline.
    Dynamically routes between Fast Path, Normal Path, and Hard Path based on confidence.
    """
    t_start = time.perf_counter()
    sh, sw = search_img.shape
    search_cx, search_cy = sw / 2.0, sh / 2.0

    # Step 1: Pre-filtering & Physical Normalization
    search_proc = cv2.GaussianBlur(search_img, (3, 3), 0.5)
    ref_proc = cv2.GaussianBlur(ref_img, (3, 3), 0.5)

    search_norm = normalize_intensity(search_proc)
    search_grad, search_ang = extract_gradient_and_orientation(search_proc)

    ref_100 = cv2.resize(ref_proc, (100, 100), interpolation=cv2.INTER_AREA)
    ref_100_norm = normalize_intensity(ref_100)
    ref_100_grad, ref_100_ang = extract_gradient_and_orientation(ref_100)

    # Step 2: Fast Global Retrieval at Native Scale 1.0 (Intensity + Gradient)
    c_i = cv2.matchTemplate(search_norm, ref_100_norm, cv2.TM_CCOEFF_NORMED)
    c_g = cv2.matchTemplate(search_grad, ref_100_grad, cv2.TM_CCOEFF_NORMED)
    c_fast = 0.55 * c_i + 0.45 * c_g

    peaks_fast = extract_distinct_spatial_peaks(c_fast, top_k=6, min_dist=15)
    if len(peaks_fast) == 0:
        return search_cx, search_cy, {"status": "NO_PEAKS", "path": "FALLBACK"}

    top1 = peaks_fast[0]
    top2 = peaks_fast[1] if len(peaks_fast) > 1 else {"score": 0.0}

    s1 = top1["score"]
    s2 = top2["score"]
    delta_s = s1 - s2
    psr_val = compute_psr(c_fast, top1["x"], top1["y"])

    path_taken = "NORMAL_PATH"
    confidence_tier = "MEDIUM"
    winning_cand = None
    all_evaluated = []

    # =========================================================================
    # REPAIR / DECISION LOGIC: ROUTING THE THREE REGIMES
    # =========================================================================

    # REGIME 1: FAST PATH (Confident & Unambiguous)
    # Conditions: High absolute score, wide peak gap, and high PSR
    if s1 >= 0.88 and delta_s >= 0.10 and psr_val >= 6.5:
        path_taken = "FAST_PATH"
        confidence_tier = "HIGH"
        winning_cand = {
            "x": top1["x"] + 50.0,
            "y": top1["y"] + 50.0,
            "peak_x": top1["x"],
            "peak_y": top1["y"],
            "tw": 100,
            "th": 100,
            "scale": 1.0,
            "rot": 0.0,
            "score": s1,
            "corr_plane": c_fast,
            "grad_corr": 1.0,
            "phase_score": 1.0,
            "orient_score": 1.0,
            "consistency": 10.0,
            "dist_to_center": np.hypot(top1["x"] + 50.0 - search_cx, top1["y"] + 50.0 - search_cy)
        }

    # REGIME 2: HARD PATH (Low score or Severe Periodic Ambiguity or High Noise)
    # Conditions: Very low peak score (< 0.52) or very tight ambiguity (delta_s < 0.02)
    elif s1 < 0.52 or (s1 < 0.70 and delta_s < 0.025):
        path_taken = "HARD_PATH"
        confidence_tier = "LOW"

        # Bounded Coarse-to-Fine Scale Search & Micro-Rotation
        scales = (0.97, 1.00, 1.03)
        rotations = (-2.0, 0.0, 2.0)
        cands_hard = []

        for s in scales:
            tw = max(10, int(round(100 * s)))
            th = max(10, int(round(100 * s)))
            ref_s = cv2.resize(ref_proc, (tw, th), interpolation=cv2.INTER_AREA)

            for r in rotations:
                if abs(r) > 0.01:
                    M = cv2.getRotationMatrix2D((tw / 2.0, th / 2.0), r, 1.0)
                    ref_r = cv2.warpAffine(ref_s, M, (tw, th), borderMode=cv2.BORDER_REFLECT)
                else:
                    ref_r = ref_s

                r_norm = normalize_intensity(ref_r)
                r_grad, r_ang = extract_gradient_and_orientation(ref_r)

                c_i_s = cv2.matchTemplate(search_norm, r_norm, cv2.TM_CCOEFF_NORMED)
                c_g_s = cv2.matchTemplate(search_grad, r_grad, cv2.TM_CCOEFF_NORMED)
                c_combo_s = 0.55 * c_i_s + 0.45 * c_g_s

                peaks_s = extract_distinct_spatial_peaks(c_combo_s, top_k=3, min_dist=15)
                for p in peaks_s:
                    cands_hard.append({
                        "x": p["x"] + tw / 2.0,
                        "y": p["y"] + th / 2.0,
                        "peak_x": p["x"],
                        "peak_y": p["y"],
                        "tw": tw,
                        "th": th,
                        "scale": s,
                        "rot": r,
                        "score": p["score"],
                        "corr_plane": c_combo_s,
                        "ref_n": r_norm,
                        "ref_g": r_grad,
                        "ref_ang": r_ang
                    })

        # Deduplicate to Top-6
        cands_hard.sort(key=lambda c: c["score"], reverse=True)
        unique_hard = []
        for c in cands_hard:
            if not any(np.hypot(c["x"] - u["x"], c["y"] - u["y"]) < 12 for u in unique_hard):
                unique_hard.append(c)
            if len(unique_hard) >= 6:
                break

        # Verification on Top-6
        for c in unique_hard:
            cx, cy = c["x"], c["y"]
            tw, th = c["tw"], c["th"]
            y1 = max(0, int(round(cy - th / 2.0)))
            y2 = min(sh, int(round(cy + th / 2.0)))
            x1 = max(0, int(round(cx - tw / 2.0)))
            x2 = min(sw, int(round(cx + tw / 2.0)))

            sp_n = search_norm[y1:y2, x1:x2]
            sp_g = search_grad[y1:y2, x1:x2]
            sp_ang = search_ang[y1:y2, x1:x2]

            if sp_n.shape != (th, tw):
                sp_n = cv2.resize(sp_n, (tw, th), interpolation=cv2.INTER_AREA)
                sp_g = cv2.resize(sp_g, (tw, th), interpolation=cv2.INTER_AREA)
                sp_ang = cv2.resize(sp_ang, (tw, th), interpolation=cv2.INTER_NEAREST)

            _, _, ps = local_phase_correlation(c["ref_n"], sp_n)
            g_corr = float(np.corrcoef(c["ref_g"].ravel(), sp_g.ravel())[0, 1])
            if np.isnan(g_corr):
                g_corr = 0.0

            ang_diff = np.abs(np.arctan2(np.sin(c["ref_ang"] - sp_ang), np.cos(c["ref_ang"] - sp_ang)))
            orient_score = float(1.0 - np.mean(ang_diff) / np.pi)

            # Consistency Metric C_i
            scores_vec = np.array([c["score"], max(0.0, g_corr), max(0.0, ps), orient_score])
            mu_i = float(np.mean(scores_vec))
            std_i = float(np.std(scores_vec))
            consistency = mu_i / (std_i + 1e-4)

            dist_to_center = float(np.hypot(cx - search_cx, cy - search_cy))

            c["grad_corr"] = g_corr
            c["phase_score"] = ps
            c["orient_score"] = orient_score
            c["consistency"] = consistency
            c["dist_to_center"] = dist_to_center

            # Hierarchical Score Fusion
            c["final_score"] = (
                0.40 * c["score"] +
                0.30 * g_corr +
                0.15 * max(0.0, ps) +
                0.10 * orient_score +
                0.05 * min(1.0, consistency / 5.0)
            )

        unique_hard.sort(key=lambda c: c["final_score"], reverse=True)
        all_evaluated = unique_hard
        winning_cand = unique_hard[0]

    # REGIME 3: NORMAL PATH (Moderate Ambiguity Cascade)
    else:
        path_taken = "NORMAL_PATH"
        confidence_tier = "MEDIUM"

        cands_normal = []
        for p in peaks_fast:
            cx, cy = p["x"] + 50.0, p["y"] + 50.0
            y1, y2 = max(0, int(round(cy - 50))), min(sh, int(round(cy + 50)))
            x1, x2 = max(0, int(round(cx - 50))), min(sw, int(round(cx + 50)))

            sp_n = search_norm[y1:y2, x1:x2]
            sp_g = search_grad[y1:y2, x1:x2]
            sp_ang = search_ang[y1:y2, x1:x2]

            if sp_n.shape != (100, 100):
                sp_n = cv2.resize(sp_n, (100, 100))
                sp_g = cv2.resize(sp_g, (100, 100))
                sp_ang = cv2.resize(sp_ang, (100, 100), interpolation=cv2.INTER_NEAREST)

            # Cheap Filter: Gradient Correlation + Orientation
            g_corr = float(np.corrcoef(ref_100_grad.ravel(), sp_g.ravel())[0, 1])
            if np.isnan(g_corr):
                g_corr = 0.0

            ang_diff = np.abs(np.arctan2(np.sin(ref_100_ang - sp_ang), np.cos(ref_100_ang - sp_ang)))
            orient_score = float(1.0 - np.mean(ang_diff) / np.pi)

            cands_normal.append({
                "x": cx,
                "y": cy,
                "peak_x": p["x"],
                "peak_y": p["y"],
                "tw": 100,
                "th": 100,
                "scale": 1.0,
                "rot": 0.0,
                "score": p["score"],
                "corr_plane": c_fast,
                "ref_n": ref_100_norm,
                "sp_n": sp_n,
                "grad_corr": g_corr,
                "orient_score": orient_score,
                "filter_score": 0.50 * p["score"] + 0.30 * g_corr + 0.20 * orient_score,
                "dist_to_center": float(np.hypot(cx - search_cx, cy - search_cy))
            })

        # Filter down to Top-3 for Phase Correlation
        cands_normal.sort(key=lambda c: c["filter_score"], reverse=True)
        top3 = cands_normal[:3]

        for c in top3:
            _, _, ps = local_phase_correlation(c["ref_n"], c["sp_n"])
            scores_vec = np.array([c["score"], max(0.0, c["grad_corr"]), max(0.0, ps), c["orient_score"]])
            mu_i = float(np.mean(scores_vec))
            std_i = float(np.std(scores_vec))
            consistency = mu_i / (std_i + 1e-4)

            c["phase_score"] = ps
            c["consistency"] = consistency

            c["final_score"] = (
                0.40 * c["score"] +
                0.30 * c["grad_corr"] +
                0.15 * max(0.0, ps) +
                0.10 * c["orient_score"] +
                0.05 * min(1.0, consistency / 5.0)
            )

        top3.sort(key=lambda c: c["final_score"], reverse=True)
        all_evaluated = top3
        winning_cand = top3[0]

    # =========================================================================
    # CONDITIONAL PERIODIC AMBIGUITY & CENTER TIE-BREAKER
    # =========================================================================
    # Rule: Structural Evidence > Center Prior. Center rule invoked ONLY on true ties.
    is_ambiguous = False
    if len(all_evaluated) > 1 and winning_cand is not None:
        c1 = all_evaluated[0]
        c2 = all_evaluated[1]
        score_diff = c1["final_score"] - c2["final_score"]

        # If candidates are separated by typical lattice pitch (e.g. 15-80 px) with nearly identical scores
        cand_dist = np.hypot(c1["x"] - c2["x"], c1["y"] - c2["y"])
        if score_diff < 0.020 and 12.0 <= cand_dist <= 120.0:
            is_ambiguous = True
            ambiguity_pool = [c for c in all_evaluated if (c1["final_score"] - c["final_score"]) < 0.020]
            winning_cand = min(ambiguity_pool, key=lambda c: c["dist_to_center"])

    # Step 6: Subpixel 2D Paraboloid Refinement
    sub_x, sub_y = subpixel_refine_2d(winning_cand["corr_plane"], winning_cand["peak_x"], winning_cand["peak_y"])
    final_x = float(sub_x + winning_cand["tw"] / 2.0)
    final_y = float(sub_y + winning_cand["th"] / 2.0)

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    meta = {
        "x": round(final_x, 2),
        "y": round(final_y, 2),
        "scale": round(winning_cand.get("scale", 1.0), 4),
        "rotation": round(winning_cand.get("rot", 0.0), 2),
        "score": round(winning_cand.get("final_score", winning_cand["score"]), 4),
        "raw_score": round(winning_cand["score"], 4),
        "psr": round(psr_val, 2),
        "path_taken": path_taken,
        "confidence_tier": confidence_tier,
        "is_ambiguous": is_ambiguous,
        "consistency": round(winning_cand.get("consistency", 0.0), 2),
        "latency_ms": round(elapsed_ms, 2)
    }

    return final_x, final_y, meta


def perform_drift_sense_localization(ref_img: np.ndarray, search_img: np.ndarray, verbose: bool = False) -> tuple[float, float, dict]:
    """Default competition entrypoint."""
    return perform_adaptive_localization(ref_img, search_img, verbose=verbose)


def main():
    parser = argparse.ArgumentParser(description="Drift-Sense++ Adaptive Structural Registration")
    parser.add_argument("--reference", type=str, required=True, help="Path to 100x reference image")
    parser.add_argument("--search", type=str, required=True, help="Path to 10x search image")
    parser.add_argument("--verbose", action="store_true", help="Output detailed JSON metadata")
    args = parser.parse_args()

    ref_img = cv2.imread(args.reference, cv2.IMREAD_GRAYSCALE)
    search_img = cv2.imread(args.search, cv2.IMREAD_GRAYSCALE)

    if ref_img is None or search_img is None:
        print("Error: Invalid image path.", file=sys.stderr)
        sys.exit(1)

    x, y, meta = perform_adaptive_localization(ref_img, search_img, verbose=args.verbose)

    if args.verbose:
        print(json.dumps(meta, indent=2))

    print(f"({x:.2f}, {y:.2f})")


if __name__ == "__main__":
    main()
