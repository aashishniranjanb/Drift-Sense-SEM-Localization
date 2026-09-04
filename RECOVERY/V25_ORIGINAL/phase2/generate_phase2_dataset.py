import os
import sys
import argparse
import numpy as np
import cv2
import pandas as pd

# Import layout generators from root dataset_generator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_generator import generate_finfet_layout, generate_dram_layout, apply_sem_acquisition_effects

def apply_aggressive_degradation(img_patch, seed=42):
    """
    Applies aggressive SEM acquisition degradation for Set B:
    Heavy charging, defocus blur, low electron dose (shot noise), readout noise, contrast drift.
    """
    rng = np.random.RandomState(seed)
    blur_sigma = rng.uniform(0.8, 1.4)
    dose_lambda = rng.uniform(50.0, 100.0)
    gaussian_noise_std = rng.uniform(0.02, 0.05)
    charging_std = rng.uniform(0.05, 0.15)
    edge_factor = rng.uniform(0.15, 0.30)
    
    return apply_sem_acquisition_effects(
        img_patch,
        blur_sigma=blur_sigma,
        dose_lambda=dose_lambda,
        gaussian_noise_std=gaussian_noise_std,
        edge_factor=edge_factor,
        charging_std=charging_std,
        seed=seed
    )

def generate_phase2_synthetic_data(output_dir, n_set_a=70, n_set_b=70, n_set_c=40, seed=200):
    """
    Generates standardized 180-case Phase 2 synthetic test benchmark:
      - Set A: 70 Nominal pairs (Reference Present)
      - Set B: 70 Aggressively Degraded pairs (Reference Present)
      - Set C: 40 Same-Architecture Absent pairs (Reference Absent, Hard Negatives)
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "reference"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "search"), exist_ok=True)
    
    rng = np.random.RandomState(seed)
    
    print("Generating synthetic layout canvases (DRAM & FinFET)...")
    finfet_canvas = generate_finfet_layout(seed=seed)
    dram_canvas = generate_dram_layout(seed=seed + 1)
    
    metadata = []
    
    total_pairs = n_set_a + n_set_b + n_set_c
    
    for i in range(total_pairs):
        pair_id = f"pair_{i:03d}"
        
        # Determine set assignment
        if i < n_set_a:
            set_type = "SetA"
            present = 1
            is_degraded = False
        elif i < n_set_a + n_set_b:
            set_type = "SetB"
            present = 1
            is_degraded = True
        else:
            set_type = "SetC"
            present = 0
            is_degraded = rng.rand() > 0.5
            
        # Alternate layout architectures
        is_dram = (i % 2 == 0)
        canvas = dram_canvas if is_dram else finfet_canvas
        
        # Reference center
        ref_cx = rng.randint(2000, 8000)
        ref_cy = rng.randint(2000, 8000)
        
        # Crop reference (1000x1000)
        ref_patch = canvas[ref_cy-500:ref_cy+500, ref_cx-500:ref_cx+500].copy()
        
        # Random scale & rotation
        scale = float(rng.uniform(8.0, 12.0))
        theta = float(rng.uniform(-5.0, 5.0))
        
        # Search center offset
        dx_canvas = rng.randint(-1500, 1500)
        dy_canvas = rng.randint(-1500, 1500)
        
        if present:
            search_cx = ref_cx + dx_canvas
            search_cy = ref_cy + dy_canvas
            search_canvas = canvas
        else:
            # Set C: Same architecture canvas, but offset to a non-overlapping region
            # (presents identical cell periodicity but absent target structure)
            search_cx = (ref_cx + 4000) % 8000 + 1000
            search_cy = (ref_cy + 4000) % 8000 + 1000
            search_canvas = canvas
            
        # Crop larger region for search to support scale & rotation
        crop_sz = int(round(1000 * scale * 1.5))
        y1 = max(0, int(search_cy - crop_sz // 2))
        y2 = min(search_canvas.shape[0], int(search_cy + crop_sz // 2))
        x1 = max(0, int(search_cx - crop_sz // 2))
        x2 = min(search_canvas.shape[1], int(search_cx + crop_sz // 2))
        
        search_patch_large = search_canvas[y1:y2, x1:x2].copy()
        
        # Pad if edge exceeded
        pad_y1 = max(0, -int(search_cy - crop_sz // 2))
        pad_y2 = max(0, int(search_cy + crop_sz // 2) - search_canvas.shape[0])
        pad_x1 = max(0, -int(search_cx - crop_sz // 2))
        pad_x2 = max(0, int(search_cx + crop_sz // 2) - search_canvas.shape[1])
        if pad_y1 > 0 or pad_y2 > 0 or pad_x1 > 0 or pad_x2 > 0:
            search_patch_large = cv2.copyMakeBorder(search_patch_large, pad_y1, pad_y2, pad_x1, pad_x2, cv2.BORDER_REFLECT)
            
        # Apply rotation
        large_h, large_w = search_patch_large.shape[:2]
        center = (large_w / 2.0, large_h / 2.0)
        M = cv2.getRotationMatrix2D(center, theta, 1.0)
        search_rotated = cv2.warpAffine(search_patch_large, M, (large_w, large_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # Crop to exact target size before resizing
        target_sz = int(round(1000 * scale))
        sy1 = int(center[1] - target_sz // 2)
        sx1 = int(center[0] - target_sz // 2)
        search_cropped = search_rotated[sy1:sy1+target_sz, sx1:sx1+target_sz]
        
        # Resize to 1000x1000
        search_resized = cv2.resize(search_cropped, (1000, 1000), interpolation=cv2.INTER_AREA)
        
        # Compute exact ground truth center (gt_x, gt_y) in search image
        if present:
            rel_ref_x = ref_cx - search_cx
            rel_ref_y = ref_cy - search_cy
            pt = np.array([rel_ref_x + large_w/2.0, rel_ref_y + large_h/2.0, 1.0])
            pt_rot = M @ pt
            x_in_target = pt_rot[0] - sx1
            y_in_target = pt_rot[1] - sy1
            gt_x = (x_in_target / target_sz) * 1000.0
            gt_y = (y_in_target / target_sz) * 1000.0
        else:
            gt_x = -1.0
            gt_y = -1.0
            
        # Apply acquisition effects
        if is_degraded:
            ref_noisy = (apply_aggressive_degradation(ref_patch, seed=i) * 255.0).astype(np.uint8)
            search_noisy = (apply_aggressive_degradation(search_resized, seed=i+1000) * 255.0).astype(np.uint8)
        else:
            ref_noisy = (apply_sem_acquisition_effects(ref_patch, seed=i) * 255.0).astype(np.uint8)
            search_noisy = (apply_sem_acquisition_effects(search_resized, seed=i+1000) * 255.0).astype(np.uint8)
            
        ref_path = os.path.relpath(os.path.join(output_dir, "reference", f"{pair_id}.png"), output_dir)
        search_path = os.path.relpath(os.path.join(output_dir, "search", f"{pair_id}.png"), output_dir)
        
        cv2.imwrite(os.path.join(output_dir, "reference", f"{pair_id}.png"), ref_noisy)
        cv2.imwrite(os.path.join(output_dir, "search", f"{pair_id}.png"), search_noisy)
        
        metadata.append({
            "pair_id": pair_id,
            "set_type": set_type,
            "reference_path": ref_path,
            "search_path": search_path,
            "gt_x": gt_x,
            "gt_y": gt_y,
            "gt_theta": theta if present else 0.0,
            "gt_scale": scale if present else 0.0,
            "gt_found": present
        })
        
    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(output_dir, "pairs.csv"), index=False)
    print(f"Dataset generation complete. 180 total pairs (70 A, 70 B, 40 C) saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 Dataset Generator")
    parser.add_argument("--output-dir", default="data/phase2_dev", help="Output directory")
    parser.add_argument("--set-a", type=int, default=70, help="Set A nominal count")
    parser.add_argument("--set-b", type=int, default=70, help="Set B degraded count")
    parser.add_argument("--set-c", type=int, default=40, help="Set C absent count")
    args = parser.parse_args()
    generate_phase2_synthetic_data(args.output_dir, args.set_a, args.set_b, args.set_c)
