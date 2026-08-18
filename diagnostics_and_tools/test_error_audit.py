"""
Fast Error Audit Script for V4 and Baselines
"""
import os
import pandas as pd
import numpy as np
import cv2

from inference import perform_proposed_localization
from hf_space.baseline_solution.zncc import zncc_match

MANIFEST_PATH = "data/benchmark_120/manifest.csv"
df = pd.read_csv(MANIFEST_PATH)

print(f"Auditing error distribution across {len(df)} samples...")

bins = [0, 1, 2, 3, 4, 5, 10, 25, 50, 100, 10000]
bin_labels = ["0-1 px", "1-2 px", "2-3 px", "3-4 px", "4-5 px", "5-10 px", "10-25 px", "25-50 px", "50-100 px", ">100 px"]

errors_v4 = []
errors_zncc = []

for idx, row in df.iterrows():
    ref_img = cv2.imread(row["reference_path"], cv2.IMREAD_GRAYSCALE)
    search_img = cv2.imread(row["search_path"], cv2.IMREAD_GRAYSCALE)
    gt_x, gt_y = float(row["gt_x"]), float(row["gt_y"])

    x_v4, y_v4, _ = perform_proposed_localization(ref_img, search_img, verbose=False)
    err_v4 = float(np.hypot(x_v4 - gt_x, y_v4 - gt_y))
    errors_v4.append(err_v4)

    res_zncc = zncc_match(ref_img, search_img)
    err_zncc = float(np.hypot(res_zncc["x"] - gt_x, res_zncc["y"] - gt_y))
    errors_zncc.append(err_zncc)

arr_v4 = np.array(errors_v4)
arr_zncc = np.array(errors_zncc)

counts_v4, _ = np.histogram(arr_v4, bins=bins)
counts_zncc, _ = np.histogram(arr_zncc, bins=bins)

df_hist = pd.DataFrame({
    "Bin": bin_labels,
    "ZNCC Counts": counts_zncc,
    "ZNCC %": np.round(counts_zncc / len(df) * 100, 2),
    "V4 Counts": counts_v4,
    "V4 %": np.round(counts_v4 / len(df) * 100, 2)
})

print("\nExact Error Distribution Histogram:")
print(df_hist.to_string(index=False))

print("\nAnalysis of V4 exact error values <= 10 px:")
small_errs = arr_v4[arr_v4 <= 10.0]
print(f"Total samples with error <= 10 px: {len(small_errs)} / {len(df)} ({len(small_errs)/len(df)*100:.2f}%)")
print("Small error list (sorted):")
print(np.round(np.sort(small_errs), 3))

print("\nSmallest error above 5 px in V4:")
above_5 = arr_v4[arr_v4 > 5.0]
print(f"Minimum error among incorrect cases: {np.min(above_5):.2f} px")
print(f"Top 5 smallest errors above 5 px: {np.round(np.sort(above_5)[:5], 2)} px")
