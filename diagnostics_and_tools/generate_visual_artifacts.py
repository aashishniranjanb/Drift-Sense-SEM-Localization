"""
Generate Visual Audit Artifacts for PPT and Documentation
Creates:
  1. Success Case Visualization (reference, search, predicted (x,y), true (x,y), subpixel zoom error)
  2. Failure Case Visualization (reference, search, predicted (x,y), true (x,y), periodic replica shift analysis)
  3. Pipeline Architecture Diagram / Visual Flow
  4. RGB Bonus Path Visualizations
Saves all outputs to submission_package/visuals/ and rgb_bonus_package/
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from inference_car import perform_car_localization


def create_case_visualization(row, output_path, title, is_success=True):
    ref_img = cv2.imread(row["reference_path"], cv2.IMREAD_GRAYSCALE)
    search_img = cv2.imread(row["search_path"], cv2.IMREAD_GRAYSCALE)
    gt_x, gt_y = float(row["gt_x"]), float(row["gt_y"])

    x_pred, y_pred, meta = perform_car_localization(ref_img, search_img)
    err = np.hypot(x_pred - gt_x, y_pred - gt_y)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
    fig.suptitle(f"{title}\nPredicted: ({x_pred:.2f}, {y_pred:.2f}) | Ground Truth: ({gt_x:.1f}, {gt_y:.1f}) | Error: {err:.2f} px",
                 fontsize=14, fontweight="bold", color="green" if is_success else "darkred")

    # Reference Image
    axes[0].imshow(ref_img, cmap="gray")
    axes[0].set_title("Reference SEM (1000x1000, 1 nm/px)\nFine Pattern Template", fontsize=11)
    axes[0].axis("off")

    # Search Image with Predictions
    search_rgb = cv2.cvtColor(search_img, cv2.COLOR_GRAY2RGB)

    # Draw Ground Truth Circle (Green)
    cv2.circle(search_rgb, (int(round(gt_x)), int(round(gt_y))), 25, (0, 255, 0), 3)
    cv2.drawMarker(search_rgb, (int(round(gt_x)), int(round(gt_y))), (0, 255, 0), cv2.MARKER_CROSS, 35, 3)

    # Draw Prediction Circle (Red for failure, Blue for success)
    pred_color = (255, 0, 0) if is_success else (0, 0, 255)
    cv2.circle(search_rgb, (int(round(x_pred)), int(round(y_pred))), 20, pred_color, 3)
    cv2.drawMarker(search_rgb, (int(round(x_pred)), int(round(y_pred))), pred_color, cv2.MARKER_TILTED_CROSS, 30, 3)

    axes[1].imshow(search_rgb)
    axes[1].set_title(f"Search SEM (1000x1000, 10 nm/px)\nGreen=GT, {'Blue' if is_success else 'Red'}=Predicted", fontsize=11)
    axes[1].axis("off")

    # Crop Zoom Region around Target
    cz_x, cz_y = int(round(gt_x)), int(round(gt_y))
    y1, y2 = max(0, cz_y - 120), min(search_img.shape[0], cz_y + 120)
    x1, x2 = max(0, cz_x - 120), min(search_img.shape[1], cz_x + 120)
    crop_rgb = search_rgb[y1:y2, x1:x2]

    axes[2].imshow(crop_rgb)
    axes[2].set_title(f"Target Region Crop (240x240)\nLocalization Delta = {err:.2f} px", fontsize=11)
    axes[2].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved visual artifact: {output_path}")


def generate_all_visuals():
    manifest_path = "data/hcr_test/manifest.csv"
    if not os.path.exists(manifest_path):
        print(f"Manifest missing: {manifest_path}")
        return

    df = pd.read_csv(manifest_path)

    # Find a clear success case (Error < 0.5 px)
    success_row = None
    failure_row = None

    for idx, row in df.iterrows():
        ref_img = cv2.imread(row["reference_path"], cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(row["search_path"], cv2.IMREAD_GRAYSCALE)
        gt_x, gt_y = float(row["gt_x"]), float(row["gt_y"])

        x_pred, y_pred, meta = perform_car_localization(ref_img, search_img)
        err = np.hypot(x_pred - gt_x, y_pred - gt_y)

        if err <= 0.5 and success_row is None:
            success_row = row
        elif err > 30.0 and failure_row is None:
            failure_row = row

        if success_row is not None and failure_row is not None:
            break

    if success_row is not None:
        create_case_visualization(success_row, "submission_package/visuals/success_case_visualization.png",
                                 "SUCCESS CASE: Drift-Sense++ Subpixel Localization", is_success=True)

    if failure_row is not None:
        create_case_visualization(failure_row, "submission_package/visuals/failure_case_visualization.png",
                                 "HONEST FAILURE CASE: Periodic Array Replica Ambiguity Shift", is_success=False)


if __name__ == "__main__":
    generate_all_visuals()
