#!/usr/bin/env python3
"""Standalone localization inference — Component 2 compatibility script.

    python inference.py --reference <reference.png> --search <search.png>

Prints the predicted centre of the reference pattern inside the search image:

    x=<float>
    y=<float>

This is NOT the official Phase 2 scoring entry point (that is register.py, which
takes --input pairs.csv --output predictions.csv). inference.py drives the exact
same internal localization engine on a single pair, for the standalone
reference/search interface required of the GitHub repository.

Runs without manual edits. No network. Deterministic.
"""
import argparse
import os
import sys

import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "runtime", "src"))

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
cv2.setNumThreads(1)

from utils import is_true_rgb                              # noqa: E402
import pipeline                                            # noqa: E402
import rgb_branch                                          # noqa: E402
from pose_estimator import refine_pose_v39                 # noqa: E402


def localize(reference_path, search_path):
    ref_c = cv2.imread(reference_path, cv2.IMREAD_COLOR)
    srch_c = cv2.imread(search_path, cv2.IMREAD_COLOR)
    if ref_c is None:
        raise FileNotFoundError(f"cannot read reference image: {reference_path}")
    if srch_c is None:
        raise FileNotFoundError(f"cannot read search image: {search_path}")

    if is_true_rgb(ref_c):
        p = rgb_branch.run_rgb_localization(ref_c, srch_c)
        return float(p["x"]), float(p["y"])

    g_ref = cv2.cvtColor(ref_c, cv2.COLOR_BGR2GRAY)
    g_srch = cv2.cvtColor(srch_c, cv2.COLOR_BGR2GRAY)
    res = pipeline.localize_grayscale(g_ref, g_srch)
    # standalone localizer: always report a coordinate — the best structural
    # candidate's centre, gate-independent, then V39-refined.
    bx, by = res["raw_x"], res["raw_y"]
    try:
        rx, ry, _, _, _ = refine_pose_v39(g_ref, g_srch, bx, by,
                                          res["est_theta"], res["est_scale"], max_displacement_px=1.0)
        return float(rx), float(ry)
    except Exception:
        return float(bx), float(by)


def main():
    ap = argparse.ArgumentParser(description="Standalone reference/search localizer")
    ap.add_argument("--reference", required=True, help="path to the reference image")
    ap.add_argument("--search", required=True, help="path to the search image")
    a = ap.parse_args()
    x, y = localize(a.reference, a.search)
    print(f"x={x:.2f}")
    print(f"y={y:.2f}")


if __name__ == "__main__":
    main()
