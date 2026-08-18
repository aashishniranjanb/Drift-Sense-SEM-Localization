"""
Generate Reproducible 120-Case Benchmark Dataset
Contains 30 Easy, 30 Medium, 30 Hard, and 30 Adversarial samples.
"""

import os
import sys
import csv
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_space"))

from hf_space.src.pipeline import DIFFICULTY_PRESETS, generate_sample, GenerationParams


def main():
    out_dir = "data/benchmark_120"
    os.makedirs(os.path.join(out_dir, "reference"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "search"), exist_ok=True)

    manifest_csv = os.path.join(out_dir, "manifest.csv")
    fieldnames = [
        "id", "difficulty", "architecture", "reference_path", "search_path",
        "gt_x", "gt_y", "gt_box_x", "gt_box_y", "gt_box_w", "gt_box_h",
        "scale", "rotation_deg", "translation_x_px", "translation_y_px", "seed"
    ]

    difficulties = ["Easy", "Medium", "Hard", "Adversarial"]
    samples_per_diff = 30
    architectures = ["dram_1x", "dram_dense", "finfet_10nm", "finfet_7nm"]

    with open(manifest_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        sample_id = 0
        for diff in difficulties:
            print(f"Generating 30 samples for [{diff}] difficulty...")
            params = DIFFICULTY_PRESETS[diff]

            for i in range(samples_per_diff):
                seed = 10000 + sample_id * 17
                rng = np.random.default_rng(seed)
                arch = architectures[i % len(architectures)]

                sample = generate_sample(arch, rng, params)

                ref_path = os.path.join(out_dir, "reference", f"{sample_id:04d}.png")
                search_path = os.path.join(out_dir, "search", f"{sample_id:04d}.png")

                cv2.imwrite(ref_path, sample["reference_img"])
                cv2.imwrite(search_path, sample["search_img"])

                gx0, gy0, gw, gh = sample["gt_box"]
                writer.writerow({
                    "id": sample_id,
                    "difficulty": diff,
                    "architecture": arch,
                    "reference_path": ref_path,
                    "search_path": search_path,
                    "gt_x": sample["gt_x"],
                    "gt_y": sample["gt_y"],
                    "gt_box_x": gx0,
                    "gt_box_y": gy0,
                    "gt_box_w": gw,
                    "gt_box_h": gh,
                    "scale": sample["scale"],
                    "rotation_deg": sample["rotation_deg"],
                    "translation_x_px": sample["translation_x_px"],
                    "translation_y_px": sample["translation_y_px"],
                    "seed": seed
                })
                sample_id += 1

    print(f"\nSuccessfully generated 120-case benchmark dataset in '{out_dir}'!")


if __name__ == "__main__":
    main()
