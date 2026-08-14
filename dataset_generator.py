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


def generate_finfet_layout(height=4000, width=4000, seed=42):
    """
    Generates a large synthetic FinFET chip layout canvas with macro functional blocks,
    power grid straps, gate cuts, fin line ends, and contact via arrays.
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
    for px in range(200, width, 800):
        canvas[:, px:px+60] = 0.95
        for py in range(100, height, 200):
            cv2.circle(canvas, (px+30, py), 18, 1.0, -1)

    for py in range(300, height, 1000):
        canvas[py:py+50, :] = 0.90

    # 5. Functional Macro Blocks & Gate Cuts
    macro_size = 600
    for my in range(0, height, macro_size):
        for mx in range(0, width, macro_size):
            num_cuts = rng.randint(3, 8)
            for _ in range(num_cuts):
                cx = mx + rng.randint(1, 15) * fin_pitch
                cy = my + rng.randint(1, 4) * gate_pitch
                canvas[cy:cy+gate_width, cx:cx+fin_pitch] = 0.10

            vx = mx + rng.randint(2, 14) * fin_pitch + fin_width // 2
            vy = my + rng.randint(1, 4) * gate_pitch + gate_width // 2
            cv2.circle(canvas, (vx, vy), 14, 0.98, -1)

    ler = rng.normal(0, 0.02, (height, width)).astype(np.float32)
    return np.clip(canvas + ler, 0.0, 1.0)


def generate_dram_layout(height=4000, width=4000, seed=42):
    """
    Generates a large synthetic DRAM chip array layout canvas with wordlines, bitlines,
    storage node capacitor contacts, sense-amp regions, and wordline driver straps.
    """
    rng = np.random.RandomState(seed)
    canvas = np.full((height, width), 0.15, dtype=np.float32)

    # 1. Wordlines (horizontal)
    wl_pitch = 40
    wl_width = 16
    for y in range(0, height, wl_pitch):
        canvas[y:y+wl_width, :] = 0.50

    # 2. Bitlines (vertical)
    bl_pitch = 40
    bl_width = 14
    for x in range(0, width, bl_pitch):
        canvas[:, x:x+bl_width] = 0.50

    # 3. Storage node contacts (vias at grid points)
    radius = 9
    for y in range(wl_pitch // 2, height, wl_pitch):
        for x in range(bl_pitch // 2, width, bl_pitch):
            cv2.circle(canvas, (x, y), radius, 0.85, -1)

    # 4. DRAM Block Boundaries & Power / Strap Networks
    for bx in range(250, width, 750):
        canvas[:, bx:bx+50] = 0.95

    for by in range(400, height, 800):
        canvas[by:by+40, :] = 0.90

    macro_h, macro_w = 400, 400
    for my in range(0, height, macro_h):
        for mx in range(0, width, macro_w):
            dummy_x = mx + rng.randint(2, 8) * bl_pitch + bl_pitch // 2
            dummy_y = my + rng.randint(2, 8) * wl_pitch + wl_pitch // 2
            cv2.circle(canvas, (dummy_x, dummy_y), 15, 0.98, -1)

    ler = rng.normal(0, 0.02, (height, width)).astype(np.float32)
    return np.clip(canvas + ler, 0.0, 1.0)


def apply_sem_acquisition_effects(image, blur_sigma, dose_lambda, gaussian_noise_std, edge_factor, charging_std, seed):
    """
    Applies independent physical SEM image formation model:
    1. Secondary electron edge brightening
    2. Beam PSF defocus (Gaussian blur)
    3. Electron Poisson shot noise (electron dose lambda)
    4. Gaussian detector/amplifier noise
    5. Charging scanline artifact
    """
    rng = np.random.RandomState(seed)
    img = image.copy().astype(np.float32)

    # 1. Secondary electron edge brightening
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)
    if grad_mag.max() > 0:
        img = img + edge_factor * (grad_mag / grad_mag.max())
    img = np.clip(img, 0.0, 1.0)

    # 2. Gaussian blur (beam PSF)
    if blur_sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur_sigma)

    # 3. Secondary electron Poisson shot noise
    if dose_lambda > 0:
        scaled = np.clip(img * dose_lambda, 0.0, None)
        noisy_poisson = rng.poisson(scaled).astype(np.float32) / dose_lambda
        img = noisy_poisson

    # 4. Gaussian detector noise
    if gaussian_noise_std > 0:
        gauss = rng.normal(0, gaussian_noise_std, img.shape).astype(np.float32)
        img = img + gauss

    # 5. Scanline / charging artifact (horizontal lines)
    if charging_std > 0:
        scanlines = rng.normal(0, charging_std, (img.shape[0], 1)).astype(np.float32)
        img = img + scanlines

    # Normalize to 0 - 255 uint8 range
    img_norm = np.clip(img, 0.0, 1.0)
    img_uint8 = (img_norm * 255.0).astype(np.uint8)
    return img_uint8


def generate_pair(canvas, pair_id, style, split, noise_multiplier=1.0, seed=1000):
    """
    Generates a single reference (100x) and search (10x) image pair.
    - Reference: 1000x1000 image cropped from a 100x100 physical region on canvas.
    - Search: 1000x1000 image cropped from a 1000x1000 physical region on canvas containing the reference region.
    - Transformations: Scale variation (0.95 - 1.05) and rotation (-3.0 to +3.0 deg) applied to search.
    - Independent SEM acquisitions: Ref (high dose/low noise), Search (low dose/high noise).
    """
    h_canvas, w_canvas = canvas.shape
    rng = np.random.RandomState(seed)

    margin = 400
    search_size = 1000
    ref_patch_size = 100

    sx = rng.randint(margin, w_canvas - search_size - margin)
    sy = rng.randint(margin, h_canvas - search_size - margin)

    rel_margin = 150
    rx_rel = rng.randint(rel_margin, search_size - ref_patch_size - rel_margin)
    ry_rel = rng.randint(rel_margin, search_size - ref_patch_size - rel_margin)

    x_true_base = rx_rel + ref_patch_size / 2.0
    y_true_base = ry_rel + ref_patch_size / 2.0

    ref_patch_raw = canvas[sy + ry_rel : sy + ry_rel + ref_patch_size,
                           sx + rx_rel : sx + rx_rel + ref_patch_size]
    ref_img_raw = cv2.resize(ref_patch_raw, (1000, 1000), interpolation=cv2.INTER_CUBIC)

    scale_true = rng.uniform(0.95, 1.05)
    rotation_true = rng.uniform(-3.0, 3.0)

    search_img_raw = canvas[sy : sy + search_size, sx : sx + search_size].copy()

    center = (search_size / 2.0, search_size / 2.0)
    M = cv2.getRotationMatrix2D(center, rotation_true, scale_true)
    search_img_transformed = cv2.warpAffine(search_img_raw, M, (search_size, search_size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

    point = np.array([x_true_base, y_true_base, 1.0], dtype=np.float64)
    pt_true = M.dot(point)
    x_true, y_true = float(pt_true[0]), float(pt_true[1])

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


def main():
    parser = argparse.ArgumentParser(description="Drift-Sense Dataset Generator")
    parser.add_argument("--style", type=str, default="finfet", choices=["finfet", "dram"], help="Semiconductor geometry style")
    parser.add_argument("--num_pairs", type=int, default=30, help="Number of image pairs to generate")
    parser.add_argument("--output_dir", type=str, default="data/val", help="Output directory path")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "stress"], help="Data split range")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    noise_mult_map = {"train": 1.0, "val": 1.5, "stress": 2.5}
    noise_mult = noise_mult_map.get(args.split, 1.0)

    print(f"[Dataset Generator] Generating {args.num_pairs} pairs for style={args.style}, split={args.split} in '{args.output_dir}'...")

    if args.style == "finfet":
        canvas = generate_finfet_layout(4000, 4000, seed=args.seed)
    else:
        canvas = generate_dram_layout(4000, 4000, seed=args.seed)

    records = []
    csv_path = os.path.join(args.output_dir, "ground_truth.csv")
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        records = existing_df.to_dict('records')

    for i in range(args.num_pairs):
        pair_id = f"{args.split}_{args.style}_{i:03d}"
        ref_filename = f"ref_{pair_id}.png"
        search_filename = f"search_{pair_id}.png"

        ref_path = os.path.join(images_dir, ref_filename)
        search_path = os.path.join(images_dir, search_filename)

        ref_img, search_img, x_true, y_true, s_true, r_true, noise_level = generate_pair(
            canvas, pair_id, args.style, args.split, noise_multiplier=noise_mult, seed=args.seed + i * 17
        )

        cv2.imwrite(ref_path, ref_img)
        cv2.imwrite(search_path, search_img)

        record = {
            "image_id": pair_id,
            "style": args.style,
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
