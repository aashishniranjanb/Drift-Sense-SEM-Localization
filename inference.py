"""
Drift-Sense AI: Production Localization Inference Entrypoint
Powered by Drift-Sense++ CAR (Confidence-Adaptive Candidate Ranking & Dual Subpixel Metrology) Engine.

Usage:
  python inference.py --reference <ref.png> --search <search.png> [--verbose]

Output:
  (x.xx, y.yy)

This is the standalone competition entrypoint script Applied Materials will execute on test data.
"""

import sys
import argparse
import json
import cv2
from inference_car import perform_car_localization


def main():
    parser = argparse.ArgumentParser(description="Drift-Sense Navigation-Error Recovery Inference")
    parser.add_argument("--reference", type=str, required=True, help="Path to reference SEM / optical image")
    parser.add_argument("--search", type=str, required=True, help="Path to search SEM / optical image")
    parser.add_argument("--verbose", action="store_true", help="Output detailed diagnostic metadata")
    args = parser.parse_args()

    ref_img = cv2.imread(args.reference, cv2.IMREAD_UNCHANGED)
    search_img = cv2.imread(args.search, cv2.IMREAD_UNCHANGED)

    if ref_img is None or search_img is None:
        print("Error: Invalid image path.", file=sys.stderr)
        sys.exit(1)

    # Automatic RGB -> Grayscale conversion if multi-channel
    if len(ref_img.shape) == 3 and ref_img.shape[2] == 3:
        ref_proc = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    elif len(ref_img.shape) == 3 and ref_img.shape[2] == 4:
        ref_proc = cv2.cvtColor(ref_img, cv2.COLOR_BGRA2GRAY)
    else:
        ref_proc = ref_img

    if len(search_img.shape) == 3 and search_img.shape[2] == 3:
        search_proc = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)
    elif len(search_img.shape) == 3 and search_img.shape[2] == 4:
        search_proc = cv2.cvtColor(search_img, cv2.COLOR_BGRA2GRAY)
    else:
        search_proc = search_img

    x, y, meta = perform_car_localization(ref_proc, search_proc, verbose=args.verbose)

    if args.verbose:
        print(json.dumps(meta, indent=2))

    # Competition-required single coordinate output format
    print(f"({x:.2f}, {y:.2f})")


if __name__ == "__main__":
    main()
