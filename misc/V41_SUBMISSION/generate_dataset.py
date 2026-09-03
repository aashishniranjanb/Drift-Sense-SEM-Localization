"""
Dataset Generator for Drift-Sense AI
Generates synthetic-yet-realistic SEM semiconductor image pairs (100x reference, 10x search)
with realistic semiconductor layout features (fins, gates, gate cuts, line ends, contact via arrays, power grid straps),
independent physical SEM acquisition models, structural parameters, and ground truth logging.
"""

import os
import argparse
import numpy as np
import cv2
import pandas as pd


def generate_finfet_layout(height=10000, width=10000, seed=42):
    """
    Generates a large synthetic FinFET chip layout canvas (10,000 nm x 10,000 nm)
    with macro functional blocks, power grid straps, gate cuts, fin line ends, and contact via arrays.
    """
    rng = np.random.RandomState(seed)
    canvas = np.full((height, width), 0.15, dtype=np.float32)

    # 1. Fin array (vertical)
    fin_pitch = 32
    fin_width = 12
    for x in range(0, width, fin_pitch):
        canvas[:, x:x+fin_width] = 0.50

    # 2. Gate array (horizontal)
    gate_pitch = 128
    gate_width = 32
    for y in range(0, height, gate_pitch):
        canvas[y:y+gate_width, :] = 0.70

    # 3. Fin-Gate Intersections
    for y in range(0, height, gate_pitch):
        for x in range(0, width, fin_pitch):
            canvas[y:y+gate_width, x:x+fin_width] = 0.85

    # 4. Power Grid Straps (vertical & horizontal metal distribution lines)
    for px in range(500, width, 1600):
        canvas[:, px:px+100] = 0.95
        for py in range(200, height, 400):
            cv2.circle(canvas, (px+50, py), 30, 1.0, -1)

    for py in range(600, height, 2000):
        canvas[py:py+80, :] = 0.90

    # 5. Functional Macro Blocks & Gate Cuts
    macro_size = 1000
    for my in range(0, height, macro_size):
        for mx in range(0, width, macro_size):
            num_cuts = rng.randint(5, 12)
            for _ in range(num_cuts):
                cx = mx + rng.randint(1, 25) * fin_pitch
                cy = my + rng.randint(1, 7) * gate_pitch
                canvas[cy:cy+gate_width, cx:cx+fin_pitch] = 0.10

            vx = mx + rng.randint(2, 24) * fin_pitch + fin_width // 2
            vy = my + rng.randint(1, 7) * gate_pitch + gate_width // 2
            cv2.circle(canvas, (vx, vy), 12, 1.0, -1)

    return canvas


def generate_dram_layout(height=10000, width=10000, seed=42):
    """
    Generates a synthetic DRAM chip layout canvas (10,000 nm x 10,000 nm)
    with dense 2D periodic capacitor arrays, wordlines, bitlines, and sense amplifier peripheral regions.
    """
    rng = np.random.RandomState(seed)
    canvas = np.full((height, width), 0.10, dtype=np.float32)

    # 1. Wordlines (horizontal) & Bitlines (vertical)
    wl_pitch = 48
    bl_pitch = 36
    for y in range(0, height, wl_pitch):
        canvas[y:y+16, :] = 0.40
    for x in range(0, width, bl_pitch):
        canvas[:, x:x+12] = 0.30

    # 2. Deep Trench / Capacitor Pillow Arrays
    for y in range(wl_pitch // 2, height, wl_pitch):
        for x in range(bl_pitch // 2, width, bl_pitch):
            cv2.ellipse(canvas, (x, y), (10, 14), 0, 0, 360, 0.90, -1)

    # 3. Sense Amplifier Periphery Straps
    for py in range(1000, height, 2400):
        canvas[py:py+160, :] = 0.75
    for px in range(1200, width, 2400):
        canvas[:, px:px+160] = 0.80

    return canvas


def apply_sem_acquisition_effects(img_patch, blur_sigma=0.8, dose_lambda=150.0, gaussian_noise_std=0.015,
                                  edge_factor=0.15, charging_std=0.01, seed=42):
    """
    Applies realistic physical SEM acquisition noise model:
    - E-beam Point Spread Function (PSF) blur
    - Secondary Electron (SE) Poisson Shot Noise
    - Detector Readout Noise (Gaussian)
    - High-brightness Edge Bloom Effect
    - Wafer Charging Low-Frequency Intensity Gradient
    """
    rng = np.random.RandomState(seed)
    patch = img_patch.copy()

    # 1. Edge brightening / bloom
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    patch = patch + edge_factor * mag
    patch = np.clip(patch, 0.0, 1.0)

    # 2. Charging gradient
    if charging_std > 1e-5:
        h, w = patch.shape
        x_lin = np.linspace(-1, 1, w)
        y_lin = np.linspace(-1, 1, h)
        xx, yy = np.meshgrid(x_lin, y_lin)
        charging_map = rng.uniform(-1, 1) * xx + rng.uniform(-1, 1) * yy
        patch = patch + charging_std * charging_map
        patch = np.clip(patch, 0.0, 1.0)

    # 3. E-beam defocus blur
    if blur_sigma > 0.1:
        patch = cv2.GaussianBlur(patch, (0, 0), blur_sigma)

    # 4. Secondary electron Poisson shot noise
    scaled_patch = patch * dose_lambda
    poisson_sample = rng.poisson(scaled_patch)
    noisy_patch = poisson_sample / float(dose_lambda)

    # 5. Detector readout noise (Gaussian)
    if gaussian_noise_std > 1e-5:
        g_noise = rng.normal(0.0, gaussian_noise_std, patch.shape)
        noisy_patch = noisy_patch + g_noise

    noisy_patch = np.clip(noisy_patch, 0.0, 1.0)
    return (noisy_patch * 255.0).astype(np.uint8)


def generate_pair(canvas, pair_id, style="finfet", split="val", noise_multiplier=1.0, seed=42):
    rng = np.random.RandomState(seed)
    height, width = canvas.shape

    ref_size = 1000
    search_size = 1000

    margin = 1500
    ref_center_x = rng.randint(margin + ref_size // 2, width - margin - ref_size // 2)
    ref_center_y = rng.randint(margin + ref_size // 2, height - margin - ref_size // 2)

    rx1 = ref_center_x - ref_size // 2
    ry1 = ref_center_y - ref_size // 2
    ref_img_raw = canvas[ry1:ry1+ref_size, rx1:rx1+ref_size]

    scale_true = 0.10  # 10x physical resolution difference (1 nm/px ref vs 10 nm/px search)
    rotation_true = rng.uniform(-0.5, 0.5)

    drift_max = 350.0
    dx_true = rng.uniform(-drift_max, drift_max)
    dy_true = rng.uniform(-drift_max, drift_max)

    search_center_x = ref_center_x + dx_true
    search_center_y = ref_center_y + dy_true

    sw_patch = int(round(search_size / scale_true))  # 10,000 nm
    sh_patch = int(round(search_size / scale_true))

    sx1 = int(round(search_center_x - sw_patch / 2.0))
    sy1 = int(round(search_center_y - sh_patch / 2.0))

    sx1_clamped = max(0, min(width - sw_patch, sx1))
    sy1_clamped = max(0, min(height - sh_patch, sy1))

    search_raw = canvas[sy1_clamped:sy1_clamped+sh_patch, sx1_clamped:sx1_clamped+sw_patch]
    search_img_resized = cv2.resize(search_raw, (search_size, search_size), interpolation=cv2.INTER_AREA)

    if abs(rotation_true) > 0.01:
        M = cv2.getRotationMatrix2D((search_size / 2.0, search_size / 2.0), rotation_true, 1.0)
        search_img_transformed = cv2.warpAffine(search_img_resized, M, (search_size, search_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    else:
        search_img_transformed = search_img_resized

    rel_x = (ref_center_x - (sx1_clamped + sw_patch / 2.0)) * scale_true
    rel_y = (ref_center_y - (sy1_clamped + sh_patch / 2.0)) * scale_true

    x_true = float(search_size / 2.0 + rel_x)
    y_true = float(search_size / 2.0 + rel_y)

    ref_seed = seed + 101
    ref_blur = rng.uniform(0.5, 0.8)
    ref_dose = 300.0 / noise_multiplier
    ref_gauss = rng.uniform(0.005, 0.015) * noise_multiplier
    ref_charging = 0.005 * noise_multiplier

    ref_final = apply_sem_acquisition_effects(
        ref_img_raw, blur_sigma=ref_blur, dose_lambda=ref_dose,
        gaussian_noise_std=ref_gauss, edge_factor=0.20, charging_std=ref_charging, seed=ref_seed
    )

    search_seed = seed + 202
    search_blur = rng.uniform(0.8, 1.2)
    search_dose = 80.0 / noise_multiplier
    search_gauss = rng.uniform(0.02, 0.05) * noise_multiplier
    search_charging = 0.02 * noise_multiplier

    search_final = apply_sem_acquisition_effects(
        search_img_transformed, blur_sigma=search_blur, dose_lambda=search_dose,
        gaussian_noise_std=search_gauss, edge_factor=0.12, charging_std=search_charging, seed=search_seed
    )

    return ref_final, search_final, x_true, y_true, float(scale_true), float(rotation_true), float(search_gauss)


def generate_synthetic_pair(architecture="DRAM", pair_id=0, seed=42, split="val"):
    style = architecture.lower()
    if style not in ["finfet", "dram"]:
        style = "dram"

    if style == "finfet":
        canvas = generate_finfet_layout(10000, 10000, seed=seed)
    else:
        canvas = generate_dram_layout(10000, 10000, seed=seed)

    ref_img, search_img, x_true, y_true, s_true, r_true, noise_level = generate_pair(
        canvas, f"{split}_{style}_{pair_id:03d}", style=style, split=split, noise_multiplier=1.0, seed=seed
    )

    return {
        "reference": ref_img,
        "search": search_img,
        "gt_x": x_true,
        "gt_y": y_true,
        "scale_true": s_true,
        "rotation_true": r_true,
    }


def main():
    parser = argparse.ArgumentParser(description="Drift-Sense Dataset Generator")
    parser.add_argument("--style", "--architecture", type=str, default="finfet", choices=["finfet", "dram", "FINFET", "DRAM"], help="Semiconductor geometry style")
    parser.add_argument("--num_pairs", "--num-pairs", type=int, default=30, help="Number of image pairs to generate")
    parser.add_argument("--output_dir", "--output-dir", type=str, default="data/val", help="Output directory path")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "stress"], help="Data split range")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    style = args.style.lower()
    num_pairs = getattr(args, "num_pairs", getattr(args, "num_pairs", 30))
    output_dir = getattr(args, "output_dir", getattr(args, "output_dir", "data/val"))

    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    noise_mult_map = {"train": 1.0, "val": 1.5, "stress": 2.5}
    noise_mult = noise_mult_map.get(args.split, 1.0)

    print(f"[Dataset Generator] Generating {num_pairs} pairs for style={style}, split={args.split} in '{output_dir}'...")

    if style == "finfet":
        canvas = generate_finfet_layout(10000, 10000, seed=args.seed)
    else:
        canvas = generate_dram_layout(10000, 10000, seed=args.seed)

    records = []
    csv_path = os.path.join(output_dir, "ground_truth.csv")
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        records = existing_df.to_dict('records')

    for i in range(num_pairs):
        pair_id = f"{args.split}_{style}_{i:03d}"
        ref_filename = f"ref_{pair_id}.png"
        search_filename = f"search_{pair_id}.png"

        ref_path = os.path.join(images_dir, ref_filename)
        search_path = os.path.join(images_dir, search_filename)

        ref_img, search_img, x_true, y_true, s_true, r_true, noise_level = generate_pair(
            canvas, pair_id, style, args.split, noise_multiplier=noise_mult, seed=args.seed + i * 17
        )

        cv2.imwrite(ref_path, ref_img)
        cv2.imwrite(search_path, search_img)

        record = {
            "image_id": pair_id,
            "style": style,
            "split": args.split,
            "ref_path": ref_path,
            "search_path": search_path,
            "x_true": round(x_true, 3),
            "y_true": round(y_true, 3),
            "scale_true": round(s_true, 4),
            "rotation_true": round(r_true, 2),
            "noise_level_search": round(noise_level, 4)
        }
        records.append(record)

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["image_id"], keep="last")
    df.to_csv(csv_path, index=False)
    print(f"[Dataset Generator] Successfully saved {len(df)} records to '{csv_path}'.")


if __name__ == "__main__":
    main()
