"""
register.py  —  Drift-Sense Phase 2 Final Submission (V41 Calibration)
----------------------------------------------------------------------
Usage:
    python register.py --input pairs.csv --output predictions.csv

Output columns: pair_id, x, y, theta, scale, found, score
  • found=0 → x=y=theta=scale=0.0
"""

import argparse
import os
import sys
import cv2
import numpy as np
import pandas as pd

# ── path setup ──────────────────────────────────────────────────────────────
_HERE = os.path.abspath(os.path.dirname(__file__))
for _p in (_HERE,
           os.path.join(_HERE, "V25_CHAMPIONSHIP"),
           os.path.join(_HERE, "fallbacks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from V25_CHAMPIONSHIP.v25_pipeline import run_v25_localization
from pose_fallback import perform_pose_fallback_search
from pose_refinement import refine_pose

# ── V39 pose refinement (frozen) ────────────────────────────────────────────
try:
    from V39_POSE.v39_pose_refinement import refine_pose_v39 as _v39_refine
    HAS_V39 = True
except Exception:
    HAS_V39 = False

# ── V41 calibration weights (residual-mix, score-only) ───────────────────────
# Formula:  cal_score = 0.90 * raw_score + 0.05 * top1_score + 0.05 * top1_corr
# For found=0: assigns evidence-based score instead of 0.0 to rank
#              correct-rejections above false-negatives.
_W0, _W1, _W2 = 0.90, 0.05, 0.05


def _calibrate(raw_score: float, top1_score: float, top1_corr: float) -> float:
    return float(np.clip(_W0 * raw_score + _W1 * top1_score + _W2 * top1_corr, 0.0, 1.0))


# ── RGB branch ───────────────────────────────────────────────────────────────
def _grad(img: np.ndarray) -> np.ndarray:
    f = img.astype(np.float32) / 255.0
    gx = cv2.Scharr(f, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(f, cv2.CV_32F, 0, 1)
    g = cv2.magnitude(gx, gy)
    mx = g.max()
    if mx > 1e-6:
        g /= mx
    return (g * 255).astype(np.uint8)


def _run_rgb(ref_bgr: np.ndarray, search_bgr: np.ndarray) -> dict:
    ref_y = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    search_y = cv2.cvtColor(search_bgr, cv2.COLOR_BGR2GRAY)
    pose_i = perform_pose_fallback_search(ref_y, search_y)
    corr_g = cv2.matchTemplate(
        _grad(search_y).astype(np.float32),
        _grad(pose_i["best_template"]).astype(np.float32),
        cv2.TM_CCOEFF_NORMED,
    )
    corr_union = np.maximum(pose_i["corr_plane"], corr_g)
    _, max_val, _, max_loc = cv2.minMaxLoc(corr_union)
    rx, ry, _, _ = refine_pose(
        ref_y, search_y,
        pose_i["best_scale"], pose_i["best_theta"],
        max_loc[0], max_loc[1],
        corr_union,
    )
    found = 1 if max_val > 0.4 else 0
    if not found:
        rx = ry = theta = scale = 0.0
    else:
        theta = pose_i["best_theta"]
        scale = pose_i["best_scale"]
    return {"x": float(rx), "y": float(ry), "theta": float(theta),
            "scale": float(scale), "found": found, "score": float(max_val)}


# ── grayscale branch (V25 → V28-C rejection → V41 calibration) ──────────────
def _run_gray(ref_gray: np.ndarray, search_gray: np.ndarray) -> dict:
    pred = run_v25_localization(ref_gray, search_gray, verbose=False)

    # --- V28-C hard rejection ---
    raw = float(pred.get("score", 0.0))
    if raw <= 0.873:
        pred.update({"found": 0, "x": 0.0, "y": 0.0, "theta": 0.0, "scale": 0.0})
        raw = 0.0
    else:
        pred["found"] = 1

    # --- V39 pose refinement (if available) ---
    if HAS_V39 and pred["found"] == 1:
        try:
            pred = _v39_refine(pred, ref_gray, search_gray)
        except Exception:
            pass

    # --- V41 calibration (evidence-based score) ---
    top1_score = float(pred.get("top1_score", raw))
    top1_corr  = float(pred.get("top1_corr",  raw))
    cal = _calibrate(raw, top1_score, top1_corr)
    pred["score"] = cal

    # Enforce schema: found=0 → zero geometry
    if pred["found"] == 0:
        pred.update({"x": 0.0, "y": 0.0, "theta": 0.0, "scale": 0.0})

    return pred


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Drift-Sense Phase 2 Register")
    parser.add_argument("--input",  required=True, help="Path to pairs.csv")
    parser.add_argument("--output", required=True, help="Path for predictions.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    data_dir = os.path.dirname(os.path.abspath(args.input))
    print(f"[register] {len(df)} pairs  |  data_dir={data_dir}")

    rows = []
    for idx, row in df.iterrows():
        pair_id = row["pair_id"]
        ref_path    = os.path.join(data_dir, row["reference_path"])
        search_path = os.path.join(data_dir, row["search_path"])

        ref_bgr    = cv2.imread(ref_path,    cv2.IMREAD_COLOR)
        search_bgr = cv2.imread(search_path, cv2.IMREAD_COLOR)

        if ref_bgr is None or search_bgr is None:
            pred = {"x": 0.0, "y": 0.0, "theta": 0.0, "scale": 0.0, "found": 0, "score": 0.0}
        else:
            b, g, r = cv2.split(ref_bgr)
            is_rgb = not (np.array_equal(b, g) and np.array_equal(g, r))
            if is_rgb:
                pred = _run_rgb(ref_bgr, search_bgr)
            else:
                gray_r = cv2.cvtColor(ref_bgr,    cv2.COLOR_BGR2GRAY)
                gray_s = cv2.cvtColor(search_bgr, cv2.COLOR_BGR2GRAY)
                pred   = _run_gray(gray_r, gray_s)

        rows.append({
            "pair_id": pair_id,
            "x": pred["x"], "y": pred["y"],
            "theta": pred["theta"], "scale": pred["scale"],
            "found": pred["found"], "score": pred["score"],
        })
        print(f"  [{idx+1:3d}/{len(df)}] {pair_id}  found={pred['found']}  score={pred['score']:.4f}")

    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"[register] Done → {args.output}")


if __name__ == "__main__":
    main()
