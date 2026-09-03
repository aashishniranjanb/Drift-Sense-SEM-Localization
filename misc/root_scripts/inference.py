"""
Drift-Sense AI: Phase 2 Production Localization Inference Entrypoint
Powered by Drift-Sense++ modular fallback architecture.

Usage:
  python inference.py --reference <ref.png> --search <search.png> [--verbose]

Output:
  (x.xx, y.yy)
"""

import os
import sys
import argparse
import json
import cv2

# Add production_engine to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "production_engine"))
from production_runner import run_production_localization

def main():
    parser = argparse.ArgumentParser(description="Drift-Sense Navigation-Error Recovery Inference")
    parser.add_argument("--reference", type=str, required=True, help="Path to reference SEM / optical image")
    parser.add_argument("--search", type=str, required=True, help="Path to search SEM / optical image")
    parser.add_argument("--verbose", action="store_true", help="Output detailed diagnostic metadata")
    args = parser.parse_args()

    ref_img = cv2.imread(args.reference, cv2.IMREAD_GRAYSCALE)
    search_img = cv2.imread(args.search, cv2.IMREAD_GRAYSCALE)

    if ref_img is None or search_img is None:
        print("Error: Invalid image path.", file=sys.stderr)
        sys.exit(1)

    # Run the production integration runner
    res = run_production_localization(ref_img, search_img, verbose=args.verbose)

    if args.verbose:
        print(json.dumps(res, indent=2))

    # Competition-required coordinate output format
    if res["found"] == 1:
        print(f"({res['x']:.2f}, {res['y']:.2f})")
    else:
        print("REJECTED (Target Absent)")

if __name__ == "__main__":
    main()
