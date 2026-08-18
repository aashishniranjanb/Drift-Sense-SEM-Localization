"""
Generate Full RGB Bonus Package & Artifacts
Creates synthetic RGB image pairs (reference_rgb.png, search_rgb.png),
runs RGB -> grayscale and 3-channel structural processing,
saves outputs in rgb_bonus_package/
"""

import os
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt

from inference import main as inference_cli
from inference_car import perform_car_localization


def generate_rgb_synthetic_pair():
    os.makedirs("rgb_bonus_package", exist_ok=True)
    os.makedirs("rgb_bonus_package/images", exist_ok=True)

    np.random.seed(101)

    # 1. Generate 1000x1000 RGB Reference Die
    ref_rgb = np.zeros((1000, 1000, 3), dtype=np.uint8)

    # Base semiconductor colors (Substrate, Oxide, Polysilicon lines)
    ref_rgb[:, :] = [40, 45, 50]  # Substrate dark grey

    # Draw periodic FinFET array lines with color variation
    for x in range(0, 1000, 20):
        ref_rgb[:, x:x+8, 0] = 180  # Red channel line
        ref_rgb[:, x:x+8, 1] = 120  # Green channel line
        ref_rgb[:, x:x+8, 2] = 60   # Blue channel line

    for y in range(0, 1000, 30):
        ref_rgb[y:y+10, :, 1] = 200 # Oxide layer green tint

    # Distinguishing central site
    cv2.rectangle(ref_rgb, (450, 450), (550, 550), (255, 200, 50), -1)
    cv2.circle(ref_rgb, (500, 500), 25, (50, 255, 255), -1)

    ref_path = "rgb_bonus_package/images/reference_rgb.png"
    cv2.imwrite(ref_path, ref_rgb)

    # 2. Generate 1000x1000 RGB Search Die (Compressed 10x target + background)
    search_rgb = np.random.randint(20, 50, (1000, 1000, 3), dtype=np.uint8)

    # Compress reference down to 100x100
    ref_100_rgb = cv2.resize(ref_rgb, (100, 100), interpolation=cv2.INTER_AREA)

    # Insert target at (gt_x=620, gt_y=380) -> top-left = (570, 330)
    gt_x, gt_y = 620.0, 380.0
    tl_x, tl_y = int(gt_x - 50), int(gt_y - 50)
    search_rgb[tl_y:tl_y+100, tl_x:tl_x+100] = ref_100_rgb

    # Add Gaussian & Poisson noise to RGB channels
    noise = np.random.normal(0, 15, search_rgb.shape).astype(np.int16)
    search_noisy = np.clip(search_rgb.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    search_path = "rgb_bonus_package/images/search_rgb.png"
    cv2.imwrite(search_path, search_noisy)

    # Run CAR localization on RGB pair
    ref_gray = cv2.cvtColor(ref_rgb, cv2.COLOR_BGR2GRAY)
    search_gray = cv2.cvtColor(search_noisy, cv2.COLOR_BGR2GRAY)

    x_pred, y_pred, meta = perform_car_localization(ref_gray, search_gray)
    err = np.hypot(x_pred - gt_x, y_pred - gt_y)

    # Plot & save RGB localization visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
    fig.suptitle(f"RGB BONUS PATH: Drift-Sense++ CAR Localization\nPredicted: ({x_pred:.2f}, {y_pred:.2f}) | GT: ({gt_x:.1f}, {gt_y:.1f}) | Subpixel Error: {err:.2f} px",
                 fontsize=14, fontweight="bold", color="darkgreen")

    axes[0].imshow(cv2.cvtColor(ref_rgb, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Reference Die (RGB 1000x1000, 1 nm/px)\nHigh-Res Multi-Channel Pattern", fontsize=11)
    axes[0].axis("off")

    vis_search = search_noisy.copy()
    cv2.circle(vis_search, (int(gt_x), int(gt_y)), 25, (0, 255, 0), 3)
    cv2.drawMarker(vis_search, (int(gt_x), int(gt_y)), (0, 255, 0), cv2.MARKER_CROSS, 35, 3)
    cv2.circle(vis_search, (int(round(x_pred)), int(round(y_pred))), 20, (255, 0, 0), 3)

    axes[1].imshow(cv2.cvtColor(vis_search, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Search Die (RGB 1000x1000, 10 nm/px)\nGreen=GT, Red=Predicted", fontsize=11)
    axes[1].axis("off")

    crop = vis_search[int(gt_y)-100:int(gt_y)+100, int(gt_x)-100:int(gt_x)+100]
    axes[2].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"RGB Target Crop (200x200)\nSubpixel Error = {err:.2f} px", fontsize=11)
    axes[2].axis("off")

    plt.tight_layout()
    vis_out = "rgb_bonus_package/images/rgb_localization_result.png"
    plt.savefig(vis_out, bbox_inches="tight")
    plt.close()

    manifest_data = {
        "reference_path": ref_path,
        "search_path": search_path,
        "gt_x": gt_x,
        "gt_y": gt_y,
        "predicted_x": round(x_pred, 2),
        "predicted_y": round(y_pred, 2),
        "error_px": round(err, 2),
        "path": meta.get("path", "FAST_PATH"),
        "latency_ms": meta.get("latency_ms", 0),
    }

    with open("rgb_bonus_package/manifest.json", "w") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"Generated RGB Bonus Package in 'rgb_bonus_package/' (Error: {err:.2f} px)")


if __name__ == "__main__":
    generate_rgb_synthetic_pair()
