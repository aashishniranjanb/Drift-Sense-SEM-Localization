"""
Drift-Sense++ HCR: High-Precision Hybrid Structural Registration Engine
Multi-Anchor Candidate Retrieval + Siamese Hard-Negative Re-ranking + Metrology Verification
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import cv2
import torch

from siamese_model import MultiScaleSiameseEncoder
from anchor_consensus import select_distinctive_anchors

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "siamese_best.pt")


def normalize_intensity(image: np.ndarray) -> np.ndarray:
    img_f = image.astype(np.float32)
    p_low, p_high = np.percentile(img_f, (1, 99))
    if p_high > p_low:
        return np.clip((img_f - p_low) / (p_high - p_low), 0.0, 1.0).astype(np.float32)
    return (img_f / 255.0).astype(np.float32)


def extract_gradient_and_orientation(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    img_f = image.astype(np.float32) / 255.0 if image.dtype == np.uint8 else image.astype(np.float32)
    gx = cv2.Scharr(img_f, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(img_f, cv2.CV_32F, 0, 1)
    g_mag = cv2.magnitude(gx, gy)
    g_ang = np.arctan2(gy, gx)
    mx = g_mag.max()
    if mx > 1e-6:
        g_mag /= mx
    return g_mag.astype(np.float32), g_ang.astype(np.float32)


def extract_patch_safe(image: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
    h, w = image.shape[:2]
    half = size // 2
    x1 = int(round(cx)) - half
    y1 = int(round(cy)) - half
    x2 = x1 + size
    y2 = y1 + size

    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        padded = cv2.copyMakeBorder(image, half, half, half, half, cv2.BORDER_REFLECT)
        x1p = int(round(cx))
        y1p = int(round(cy))
        patch = padded[y1p:y1p+size, x1p:x1p+size]
    else:
        patch = image[y1:y2, x1:x2]

    if patch.shape[0] != size or patch.shape[1] != size:
        patch = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)
    return patch


def local_phase_correlation(ref_patch: np.ndarray, search_patch: np.ndarray) -> tuple[float, float, float]:
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
    return float(max_loc[0] - win_r), float(max_loc[1] - win_r), float(max_val)


def subpixel_refine_2d(corr_plane: np.ndarray, int_x: int, int_y: int) -> tuple[float, float]:
    """Fits 2D quadratic paraboloid surface; clamps subpixel offset strictly to [-0.5, 0.5] px."""
    h, w = corr_plane.shape
    if int_x < 2 or int_x >= w - 2 or int_y < 2 or int_y >= h - 2:
        return float(int_x), float(int_y)

    patch = corr_plane[int_y-2:int_y+3, int_x-2:int_x+3].astype(np.float64)
    y_coords, x_coords = np.mgrid[-2:3, -2:3]
    X_mat = np.column_stack([
        x_coords.ravel()**2, y_coords.ravel()**2,
        x_coords.ravel() * y_coords.ravel(),
        x_coords.ravel(), y_coords.ravel(), np.ones(25)
    ])
    Z_vec = patch.ravel()
    try:
        coeff, _, _, _ = np.linalg.lstsq(X_mat, Z_vec, rcond=None)
        a, b, c, d, e, _ = coeff
        denom = 4 * a * b - c**2
        if abs(denom) > 1e-6 and a < 0 and b < 0:
            dx = (c * e - 2 * b * d) / denom
            dy = (c * d - 2 * a * e) / denom
            dx = np.clip(dx, -0.5, 0.5)
            dy = np.clip(dy, -0.5, 0.5)
            return float(int_x + dx), float(int_y + dy)
    except Exception:
        pass
    return float(int_x), float(int_y)


# ─── Siamese Model Loading ───────────────────────────────────────────────

_SIAMESE_MODEL = None
_SIAMESE_DEVICE = None


def load_siamese_model(model_path: str = None) -> tuple:
    global _SIAMESE_MODEL, _SIAMESE_DEVICE
    if _SIAMESE_MODEL is not None:
        return _SIAMESE_MODEL, _SIAMESE_DEVICE

    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_path):
        _SIAMESE_DEVICE = device
        return None, device

    try:
        model = MultiScaleSiameseEncoder(local_dim=64, context_dim=64)
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        _SIAMESE_MODEL = model
        _SIAMESE_DEVICE = device
        return model, device
    except Exception:
        _SIAMESE_DEVICE = device
        return None, device


def siamese_rerank_candidates(
    model: MultiScaleSiameseEncoder,
    device: torch.device,
    ref_img: np.ndarray,
    search_img: np.ndarray,
    candidates: list[dict],
) -> list[dict]:
    if model is None or len(candidates) == 0:
        for c in candidates:
            c["neural_sim"] = 0.0
        return candidates

    ref_100 = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)
    ref_64 = cv2.resize(ref_100, (64, 64), interpolation=cv2.INTER_AREA)
    ref_128 = cv2.resize(ref_img, (128, 128), interpolation=cv2.INTER_AREA)

    ref_64_norm = normalize_intensity(ref_64)
    ref_128_norm = normalize_intensity(ref_128)

    ref_64_t = torch.from_numpy(ref_64_norm).unsqueeze(0).unsqueeze(0).to(device)
    ref_128_t = torch.from_numpy(ref_128_norm).unsqueeze(0).unsqueeze(0).to(device)

    cand_64_list = []
    cand_128_list = []
    for c in candidates:
        cx, cy = c["cx"], c["cy"]
        p64 = extract_patch_safe(search_img, cx, cy, 64)
        p128 = extract_patch_safe(search_img, cx, cy, 128)
        p64_norm = normalize_intensity(p64)
        p128_norm = normalize_intensity(p128)
        cand_64_list.append(torch.from_numpy(p64_norm).unsqueeze(0))
        cand_128_list.append(torch.from_numpy(p128_norm).unsqueeze(0))

    cand_64_batch = torch.stack(cand_64_list).to(device)
    cand_128_batch = torch.stack(cand_128_list).to(device)

    with torch.no_grad():
        z_ref = model(ref_64_t, ref_128_t)
        z_cands = model(cand_64_batch, cand_128_batch)
        similarities = torch.nn.functional.cosine_similarity(
            z_ref.expand_as(z_cands), z_cands
        )

    sims = similarities.cpu().numpy()
    for i, c in enumerate(candidates):
        c["neural_sim"] = float(sims[i])

    return candidates


# ─── Main HCR Pipeline ───────────────────────────────────────────────────

def perform_hcr_localization(
    ref_img: np.ndarray,
    search_img: np.ndarray,
    model_path: str = None,
    verbose: bool = False,
) -> tuple[float, float, dict]:
    t_start = time.perf_counter()
    sh, sw = search_img.shape
    search_cx, search_cy = sw / 2.0, sh / 2.0

    # ── Stage 0: Physical Normalization ──
    ref_100 = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)

    # Fast Intensity ZNCC Correlation (Native Match)
    c_i = cv2.matchTemplate(search_img.astype(np.float32), ref_100.astype(np.float32), cv2.TM_CCOEFF_NORMED)

    # Extract Top-10 spatial peaks with NMS
    work = c_i.copy()
    ch, cw = work.shape
    candidates = []
    for _ in range(10):
        _, max_val, _, max_loc = cv2.minMaxLoc(work)
        if max_val <= -1.0 or np.isnan(max_val):
            break
        px, py = max_loc
        cx = px + 50.0
        cy = py + 50.0
        candidates.append({
            "cx": cx, "cy": cy,
            "peak_x": px, "peak_y": py,
            "tw": 100, "th": 100,
            "scale": 1.0,
            "ncc_score": float(max_val),
            "corr_plane": c_i,
        })
        y1, y2 = max(0, py - 12), min(ch, py + 13)
        x1, x2 = max(0, px - 12), min(cw, px + 13)
        work[y1:y2, x1:x2] = -999.0

    if len(candidates) == 0:
        return search_cx, search_cy, {"status": "NO_CANDIDATES", "latency_ms": 0}

    # Evaluate candidate score gap (delta-S)
    top1_ncc = candidates[0]["ncc_score"]
    top2_ncc = candidates[1]["ncc_score"] if len(candidates) > 1 else 0.0
    delta_s = top1_ncc - top2_ncc

    # ── FAST PATH: Preserves 100% of baseline FFT accuracy when top1 is clear ──
    if delta_s >= 0.008 or top1_ncc >= 0.90:
        sub_x, sub_y = subpixel_refine_2d(c_i, candidates[0]["peak_x"], candidates[0]["peak_y"])
        final_x = float(sub_x + 50.0)
        final_y = float(sub_y + 50.0)
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return final_x, final_y, {
            "x": round(final_x, 2), "y": round(final_y, 2),
            "ncc_score": round(top1_ncc, 4),
            "delta_s": round(delta_s, 4),
            "path": "FAST_PATH",
            "latency_ms": round(elapsed_ms, 2),
            "status": "OK"
        }

    # ── AMBIGUOUS PATH: Anchor Augmentation + Siamese Re-Ranking + Metrology Verification ──
    # Augment candidate pool with Multi-Anchor Consensus candidates
    try:
        anchors = select_distinctive_anchors(ref_100, patch_size=36, stride=12, num_anchors=3)
        search_proc = cv2.GaussianBlur(search_img, (3, 3), 0.5)
        search_norm = normalize_intensity(search_proc)
        for a in anchors:
            c_anc = cv2.matchTemplate(search_norm, a["patch_int"], cv2.TM_CCOEFF_NORMED)
            _, a_val, _, a_loc = cv2.minMaxLoc(c_anc)
            pred_cx = a_loc[0] + 18.0 - a["offset_x"]
            pred_cy = a_loc[1] + 18.0 - a["offset_y"]
            if not any(np.hypot(pred_cx - c["cx"], pred_cy - c["cy"]) < 12 for c in candidates):
                px = max(0, min(cw - 1, int(round(pred_cx - 50.0))))
                py = max(0, min(ch - 1, int(round(pred_cy - 50.0))))
                candidates.append({
                    "cx": pred_cx, "cy": pred_cy,
                    "peak_x": px, "peak_y": py,
                    "tw": 100, "th": 100,
                    "scale": 1.0,
                    "ncc_score": float(c_i[py, px]) if py < ch and px < cw else 0.0,
                    "corr_plane": c_i,
                })
    except Exception:
        pass

    model, device = load_siamese_model(model_path)
    use_neural = (model is not None)

    if use_neural:
        candidates = siamese_rerank_candidates(model, device, ref_img, search_img, candidates)
        for c in candidates:
            # Neural score discriminates true site from periodic replica
            c["rerank_score"] = 0.60 * c["ncc_score"] + 0.40 * max(0.0, c["neural_sim"])
        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    else:
        for c in candidates:
            c["neural_sim"] = 0.0
            c["rerank_score"] = c["ncc_score"]

    top3 = candidates[:3]

    search_proc = cv2.GaussianBlur(search_img, (3, 3), 0.5)
    search_norm = normalize_intensity(search_proc)
    search_grad, search_ang = extract_gradient_and_orientation(search_proc)

    ref_proc = cv2.GaussianBlur(ref_img, (3, 3), 0.5)
    ref_100_proc = cv2.resize(ref_proc, (100, 100), interpolation=cv2.INTER_AREA)
    ref_100_norm = normalize_intensity(ref_100_proc)
    ref_100_grad, ref_100_ang = extract_gradient_and_orientation(ref_100_proc)

    for c in top3:
        cx, cy = c["cx"], c["cy"]
        y1, y2 = max(0, int(round(cy - 50))), min(sh, int(round(cy + 50)))
        x1, x2 = max(0, int(round(cx - 50))), min(sw, int(round(cx + 50)))

        sp_n = search_norm[y1:y2, x1:x2]
        sp_g = search_grad[y1:y2, x1:x2]

        if sp_n.shape != (100, 100):
            sp_n = cv2.resize(sp_n, (100, 100), interpolation=cv2.INTER_AREA)
            sp_g = cv2.resize(sp_g, (100, 100), interpolation=cv2.INTER_AREA)

        g_corr = float(np.corrcoef(ref_100_grad.ravel(), sp_g.ravel())[0, 1])
        if np.isnan(g_corr):
            g_corr = 0.0

        _, _, phase_score = local_phase_correlation(ref_100_norm, sp_n)

        c["grad_corr"] = g_corr
        c["phase_score"] = phase_score
        c["dist_to_center"] = float(np.hypot(cx - search_cx, cy - search_cy))

        c["final_score"] = (
            0.50 * c["ncc_score"] +
            0.35 * max(0.0, c["neural_sim"]) +
            0.10 * g_corr +
            0.05 * max(0.0, phase_score)
        )

    top3.sort(key=lambda c: c["final_score"], reverse=True)

    # Periodicity Disambiguation (Center prior strictly on score ties)
    best = top3[0]
    is_ambiguous = False

    if len(top3) >= 2:
        final_delta = top3[0]["final_score"] - top3[1]["final_score"]
        cand_dist = np.hypot(top3[0]["cx"] - top3[1]["cx"], top3[0]["cy"] - top3[1]["cy"])

        if final_delta < 0.008 and 15.0 <= cand_dist <= 120.0:
            is_ambiguous = True
            ambiguity_pool = [c for c in top3 if (top3[0]["final_score"] - c["final_score"]) < 0.008]
            best = min(ambiguity_pool, key=lambda c: c["dist_to_center"])

    # Subpixel 2D Paraboloid Refinement
    sub_x, sub_y = subpixel_refine_2d(best["corr_plane"], best["peak_x"], best["peak_y"])
    final_x = float(sub_x + 50.0)
    final_y = float(sub_y + 50.0)

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    meta = {
        "x": round(final_x, 2),
        "y": round(final_y, 2),
        "ncc_score": round(best["ncc_score"], 4),
        "neural_sim": round(best.get("neural_sim", 0.0), 4),
        "grad_corr": round(best.get("grad_corr", 0.0), 4),
        "phase_score": round(best.get("phase_score", 0.0), 4),
        "final_score": round(best.get("final_score", 0.0), 4),
        "is_ambiguous": is_ambiguous,
        "use_neural": use_neural,
        "dist_to_center": round(best.get("dist_to_center", 0.0), 2),
        "latency_ms": round(elapsed_ms, 2),
        "path": "AMBIGUOUS_PATH",
        "status": "OK",
    }

    return final_x, final_y, meta


def main():
    parser = argparse.ArgumentParser(description="Drift-Sense++ HCR Precision Inference")
    parser.add_argument("--reference", type=str, required=True)
    parser.add_argument("--search", type=str, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    ref_img = cv2.imread(args.reference, cv2.IMREAD_GRAYSCALE)
    search_img = cv2.imread(args.search, cv2.IMREAD_GRAYSCALE)

    if ref_img is None or search_img is None:
        print("Error: Invalid image path.", file=sys.stderr)
        sys.exit(1)

    x, y, meta = perform_hcr_localization(ref_img, search_img, model_path=args.model, verbose=args.verbose)

    if args.verbose:
        print(json.dumps(meta, indent=2))

    print(f"({x:.2f}, {y:.2f})")


if __name__ == "__main__":
    main()
