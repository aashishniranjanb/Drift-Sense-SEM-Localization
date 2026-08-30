import os
import sys
import numpy as np
import cv2
import pandas as pd

# Add parent directory to path to import layout generator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_generator import generate_finfet_layout, generate_dram_layout
from generate_phase2_dataset import apply_sem_acquisition_effects

def generate_adversarial_dataset(output_dir="data/adversarial", seed=300):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "reference"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "search"), exist_ok=True)
    
    rng = np.random.RandomState(seed)
    
    print("Generating layouts for adversarial suite...")
    finfet_canvas = generate_finfet_layout(seed=seed)
    dram_canvas = generate_dram_layout(seed=seed + 1)
    
    metadata = []
    
    # 10 categories, 15 pairs per category = 150 total cases
    categories = [
        "01_exact_periodic_replica",
        "02_near_periodic_replica",
        "03_phase_shifted_replica",
        "04_noise_degraded",
        "05_charging_degraded",
        "06_scale_extreme",
        "07_rotation_extreme",
        "08_absent_same_architecture",
        "09_absent_different_architecture",
        "10_combined_failure"
    ]
    
    pairs_per_cat = 15
    pair_counter = 0
    
    for cat_idx, cat in enumerate(categories):
        for case in range(pairs_per_cat):
            pair_id = f"adv_{pair_counter:03d}"
            pair_counter += 1
            
            is_dram = (case % 2 == 0)
            canvas = dram_canvas if is_dram else finfet_canvas
            
            # Default nominal parameters
            present = 1
            is_degraded = False
            blur_sigma = 0.8
            dose_lambda = 150.0
            gaussian_noise_std = 0.015
            charging_std = 0.05
            edge_factor = 0.10
            scale = float(rng.uniform(9.0, 11.0))
            theta = float(rng.uniform(-3.0, 3.0))
            dx_canvas = rng.randint(-1000, 1000)
            dy_canvas = rng.randint(-1000, 1000)
            
            # Reference center (force to highly periodic region: center of canvas block)
            ref_cx = rng.randint(3000, 7000)
            ref_cy = rng.randint(3000, 7000)
            
            # Apply category-specific parameters
            if cat == "01_exact_periodic_replica":
                # Clean, highly periodic region, no macro features
                pass
                
            elif cat == "02_near_periodic_replica":
                # Small offset in search to yield candidates very close to the center
                pitch = 36 if is_dram else 32
                dx_canvas = int(round(pitch * rng.choice([-1.5, -0.5, 0.5, 1.5])))
                dy_canvas = int(round(pitch * rng.choice([-1.5, -0.5, 0.5, 1.5])))
                
            elif cat == "03_phase_shifted_replica":
                # Sub-pixel offset in canvas space (e.g. 0.25, 0.50 of pitch)
                pitch = 36.0 if is_dram else 32.0
                dx_canvas = int(round(pitch * rng.randint(-5, 5) + 0.50))
                dy_canvas = int(round(pitch * rng.randint(-5, 5) + 0.50))
                
            elif cat == "04_noise_degraded":
                # Extreme noise
                is_degraded = True
                gaussian_noise_std = 0.08
                dose_lambda = 25.0
                blur_sigma = 1.6
                
            elif cat == "05_charging_degraded":
                # Extreme charging
                is_degraded = True
                charging_std = 0.35
                
            elif cat == "06_scale_extreme":
                # Scale outside [8.0, 12.0]
                scale = float(rng.choice([7.2, 12.8]))
                
            elif cat == "07_rotation_extreme":
                # Rotation outside [-5.0, 5.0]
                theta = float(rng.choice([-6.5, 6.5]))
                
            elif cat == "08_absent_same_architecture":
                # Absent (different region of same architecture)
                present = 0
                search_cx = (ref_cx + 4000) % 8000 + 1000
                search_cy = (ref_cy + 4000) % 8000 + 1000
                search_canvas = canvas
                
            elif cat == "09_absent_different_architecture":
                # Absent (different architecture canvas)
                present = 0
                search_cx = ref_cx
                search_cy = ref_cy
                search_canvas = finfet_canvas if is_dram else dram_canvas
                
            elif cat == "10_combined_failure":
                # Extreme scale, rotation, noise, and charging
                is_degraded = True
                scale = float(rng.choice([7.4, 12.6]))
                theta = float(rng.choice([-6.0, 6.0]))
                gaussian_noise_std = 0.07
                dose_lambda = 30.0
                charging_std = 0.30
                blur_sigma = 1.5
                
            # Crop reference
            ref_patch = canvas[ref_cy-500:ref_cy+500, ref_cx-500:ref_cx+500].copy()
            
            if present:
                search_cx = ref_cx + dx_canvas
                search_cy = ref_cy + dy_canvas
                search_canvas = canvas
                
            # Crop search region
            crop_sz = int(round(1000 * scale * 1.5))
            y1 = max(0, int(search_cy - crop_sz // 2))
            y2 = min(search_canvas.shape[0], int(search_cy + crop_sz // 2))
            x1 = max(0, int(search_cx - crop_sz // 2))
            x2 = min(search_canvas.shape[1], int(search_cx + crop_sz // 2))
            
            search_patch_large = search_canvas[y1:y2, x1:x2].copy()
            
            pad_y1 = max(0, -int(search_cy - crop_sz // 2))
            pad_y2 = max(0, int(search_cy + crop_sz // 2) - search_canvas.shape[0])
            pad_x1 = max(0, -int(search_cx - crop_sz // 2))
            pad_x2 = max(0, int(search_cx + crop_sz // 2) - search_canvas.shape[1])
            if pad_y1 > 0 or pad_y2 > 0 or pad_x1 > 0 or pad_x2 > 0:
                search_patch_large = cv2.copyMakeBorder(search_patch_large, pad_y1, pad_y2, pad_x1, pad_x2, cv2.BORDER_REFLECT)
                
            # Rotate search patch
            large_h, large_w = search_patch_large.shape[:2]
            center = (large_w / 2.0, large_h / 2.0)
            M = cv2.getRotationMatrix2D(center, theta, 1.0)
            search_rotated = cv2.warpAffine(search_patch_large, M, (large_w, large_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            
            # Crop to target size
            target_sz = int(round(1000 * scale))
            sy1 = int(center[1] - target_sz // 2)
            sx1 = int(center[0] - target_sz // 2)
            search_cropped = search_rotated[sy1:sy1+target_sz, sx1:sx1+target_sz]
            search_resized = cv2.resize(search_cropped, (1000, 1000), interpolation=cv2.INTER_AREA)
            
            # Apply degradations
            ref_degraded = apply_sem_acquisition_effects(ref_patch, blur_sigma=0.8, dose_lambda=150.0, gaussian_noise_std=0.015, edge_factor=0.10, charging_std=0.05, seed=case)
            search_degraded = apply_sem_acquisition_effects(search_resized, blur_sigma=blur_sigma, dose_lambda=dose_lambda, gaussian_noise_std=gaussian_noise_std, edge_factor=edge_factor, charging_std=charging_std, seed=case+1)
            
            # Compute GT center in search image coordinates
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
                gt_x, gt_y = 0.0, 0.0
                
            # Save files
            ref_name = f"reference/{pair_id}.png"
            search_name = f"search/{pair_id}.png"
            cv2.imwrite(os.path.join(output_dir, ref_name), (ref_degraded * 255.0).astype(np.uint8))
            cv2.imwrite(os.path.join(output_dir, search_name), (search_degraded * 255.0).astype(np.uint8))
            
            metadata.append({
                "pair_id": pair_id,
                "set_type": cat,
                "reference_path": ref_name,
                "search_path": search_name,
                "gt_x": gt_x,
                "gt_y": gt_y,
                "gt_theta": theta,
                "gt_scale": scale,
                "gt_found": present
            })
            
    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(output_dir, "pairs.csv"), index=False)
    print(f"Generated {len(df)} cases across 10 categories in {output_dir}")

if __name__ == "__main__":
    generate_adversarial_dataset()
