"""
Drift-Sense Physics Validation Engine
Automated Verification Suite for SEM Acquisition Physics, Ground-Truth Navigation Coordinates,
Scale Relationships, and Physics Manifest Provenance.
"""

import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_space"))

from hf_space.src.pipeline import GenerationParams, DIFFICULTY_PRESETS, generate_sample


def validate_single_sample(sample: dict, difficulty: str) -> list[tuple[str, bool, str]]:
    checks = []

    ref_img = sample["reference_img"]
    search_img = sample["search_img"]
    gt_x = sample["gt_x"]
    gt_y = sample["gt_y"]
    gt_box = sample["gt_box"]
    manifest = sample["manifest"]

    # Check 1: Reference dimensions
    r_h, r_w = ref_img.shape
    c1 = (r_h == 1000 and r_w == 1000)
    checks.append(("Reference dimensions (1000x1000 px)", c1, f"Actual: {r_w}x{r_h}"))

    # Check 2: Search dimensions
    s_h, s_w = search_img.shape
    c2 = (s_h == 1000 and s_w == 1000)
    checks.append(("Search dimensions (1000x1000 px)", c2, f"Actual: {s_w}x{s_h}"))

    # Check 3: 10x physical relationship
    c3 = (manifest["reference_acquisition"]["pixel_size_nm"] == 1 and
          manifest["search_acquisition"]["pixel_size_nm"] == 10)
    checks.append(("10x Scale Physical Relationship (1 nm/px vs 10 nm/px)", c3,
                   f"Ref: {manifest['reference_acquisition']['pixel_size_nm']} nm/px, Search: {manifest['search_acquisition']['pixel_size_nm']} nm/px"))

    # Check 4: Ground truth bounds inside Search FOV
    c4 = (0.0 <= gt_x <= 1000.0) and (0.0 <= gt_y <= 1000.0)
    checks.append(("Ground truth coordinates inside Search FOV", c4, f"gt_x={gt_x:.2f}, gt_y={gt_y:.2f}"))

    # Check 5: Ground truth box dimensions (100x100 nm footprint in 10 nm/px space)
    box_w, box_h = gt_box[2], gt_box[3]
    c5 = (box_w == 100 and box_h == 100)
    checks.append(("Reference footprint box size in Search image (100x100 px)", c5, f"Box: {box_w}x{box_h} px"))

    # Check 6: Reference and Search dose distinction
    dose_ref = manifest["reference_acquisition"]["dose"]
    dose_search = manifest["search_acquisition"]["dose"]
    c6 = (dose_ref > dose_search)
    checks.append(("Dose Separation (Ref Dose > Search Dose)", c6, f"Ref Dose={dose_ref}, Search Dose={dose_search}"))

    # Check 7: Manifest consistency
    c7 = (manifest["difficulty_level"] == difficulty)
    checks.append(("Physics Manifest difficulty consistency", c7, f"Manifest Diff: {manifest['difficulty_level']}"))

    return checks


def main():
    print("=" * 65)
    print("           DRIFT-SENSE PHYSICS VALIDATION SUITE           ")
    print("=" * 65)

    difficulties = ["Easy", "Medium", "Hard", "Adversarial"]
    all_passed = True

    for diff in difficulties:
        print(f"\n[Testing Preset: {diff}]")
        params = DIFFICULTY_PRESETS[diff]
        rng = np.random.default_rng(42)
        sample = generate_sample("finfet_10nm", rng, params)

        checks = validate_single_sample(sample, diff)

        for name, passed, detail in checks:
            status = "[PASS]" if passed else "[FAIL]"
            if not passed:
                all_passed = False
            print(f"  {status:6s} {name:<50s} | {detail}")

    print("\n" + "=" * 65)
    if all_passed:
        print("OVERALL PHYSICS VALIDATION STATUS: [PASS]")
    else:
        print("OVERALL PHYSICS VALIDATION STATUS: [FAIL]")
    print("=" * 65)

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
