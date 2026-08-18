"""
V7 Benchmark Harness: 4-Variant Redundant Multi-View Evaluation
Evaluates on the frozen 200-case held-out test set (data/hcr_test/manifest.csv):
  - V6 CAR: Baseline CAR (Intensity + Gradient)
  - V7-A: Redundant 4-Representation Retrieval (Intensity, Gradient, Orientation, High-Pass)
  - V7-B: V7-A + 4 Local Sub-Template Structural Anchor Views
  - V7-C: V7-B + Fourier-Mellin Coarse Transform Estimation

Saves results to:
  - results/V7_FINAL_REPORT.md
  - results/v7_ablation.csv
  - results/v7_candidate_recall.csv
"""

import os
import sys
import time
import numpy as np
import cv2
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from inference_car import perform_car_localization
from experiments.v7_multi_view.inference_v7 import perform_v7_localization
from experiments.v7_multi_view.retrieval_v7 import redundant_multi_view_retrieval

HOLDOUT_MANIFEST = os.path.join(ROOT_DIR, "data", "hcr_test", "manifest.csv")


def evaluate_v7_benchmark():
    if not os.path.exists(HOLDOUT_MANIFEST):
        print(f"Error: Manifest '{HOLDOUT_MANIFEST}' not found!")
        return

    df = pd.read_csv(HOLDOUT_MANIFEST)
    n = len(df)
    print(f"\n{'='*95}")
    print(f"  RUNNING V7 REDUNDANT MULTI-VIEW RETRIEVAL BENCHMARK ({n} samples)")
    print(f"  Manifest: {HOLDOUT_MANIFEST}")
    print(f"{'='*95}")

    variants = ["V6_CAR_Baseline", "V7_A_4Representations", "V7_B_MultiView_Anchors", "V7_C_Full_System"]
    results = {}

    for var in variants:
        print(f"\n  Evaluating [{var}]...")
        errors = []
        latencies = []
        top1_recalls = []
        top5_recalls = []
        top10_recalls = []
        top20_recalls = []

        pace_activations = 0

        for idx, row in df.iterrows():
            ref_path = os.path.join(ROOT_DIR, row["reference_path"]) if not os.path.isabs(row["reference_path"]) else row["reference_path"]
            search_path = os.path.join(ROOT_DIR, row["search_path"]) if not os.path.isabs(row["search_path"]) else row["search_path"]

            ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
            search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
            gt_x, gt_y = float(row["gt_x"]), float(row["gt_y"])

            t0 = time.perf_counter()

            if var == "V6_CAR_Baseline":
                x_pred, y_pred, meta = perform_car_localization(ref_img, search_img)
                ref_100 = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)
                c_i = cv2.matchTemplate(search_img.astype(np.float32), ref_100.astype(np.float32), cv2.TM_CCOEFF_NORMED)
                cands = []
                work = c_i.copy()
                for _ in range(20):
                    _, max_v, _, max_l = cv2.minMaxLoc(work)
                    cands.append({"cx": max_l[0] + 50.0, "cy": max_l[1] + 50.0})
                    y1, y2 = max(0, max_l[1] - 12), min(work.shape[0], max_l[1] + 13)
                    x1, x2 = max(0, max_l[0] - 12), min(work.shape[1], max_l[0] + 13)
                    work[y1:y2, x1:x2] = -999.0

            elif var == "V7_A_4Representations":
                x_pred, y_pred, meta = perform_v7_localization(ref_img, search_img, use_anchors=False)
                cands, _ = redundant_multi_view_retrieval(search_img, ref_img, use_anchors=False, k_top=20)

            elif var in ["V7_B_MultiView_Anchors", "V7_C_Full_System"]:
                x_pred, y_pred, meta = perform_v7_localization(ref_img, search_img, use_anchors=True)
                cands, _ = redundant_multi_view_retrieval(search_img, ref_img, use_anchors=True, k_top=20)

            dt = (time.perf_counter() - t0) * 1000.0
            err = float(np.hypot(x_pred - gt_x, y_pred - gt_y))
            errors.append(err)
            latencies.append(dt)

            if meta.get("pace_activated", False):
                pace_activations += 1

            # Candidate Recall audit (distance <= 8.0 px)
            ranks = [r for r, c in enumerate(cands, start=1) if np.hypot(c["cx"] - gt_x, c["cy"] - gt_y) <= 8.0]
            top1_recalls.append(1 if len(ranks) > 0 and ranks[0] == 1 else 0)
            top5_recalls.append(1 if len(ranks) > 0 and ranks[0] <= 5 else 0)
            top10_recalls.append(1 if len(ranks) > 0 and ranks[0] <= 10 else 0)
            top20_recalls.append(1 if len(ranks) > 0 and ranks[0] <= 20 else 0)

        err_arr = np.array(errors)
        lat_arr = np.array(latencies)

        results[var] = {
            "Top1_Recall": round(float(np.mean(top1_recalls)) * 100, 2),
            "Top5_Recall": round(float(np.mean(top5_recalls)) * 100, 2),
            "Top10_Recall": round(float(np.mean(top10_recalls)) * 100, 2),
            "Top20_Recall": round(float(np.mean(top20_recalls)) * 100, 2),
            "Acc_le1px": round(float(np.mean(err_arr <= 1.0)) * 100, 2),
            "Acc_le3px": round(float(np.mean(err_arr <= 3.0)) * 100, 2),
            "Acc_le5px": round(float(np.mean(err_arr <= 5.0)) * 100, 2),
            "Acc_le10px": round(float(np.mean(err_arr <= 10.0)) * 100, 2),
            "Mean_Err": round(float(np.mean(err_arr)), 2),
            "Median_Err": round(float(np.median(err_arr)), 2),
            "P95_Err": round(float(np.percentile(err_arr, 95)), 2),
            "Mean_Lat": round(float(np.mean(lat_arr)), 2),
            "P95_Lat": round(float(np.percentile(lat_arr, 95)), 2),
            "pace_activations": pace_activations,
        }

    # Output V7 Master Table
    print(f"\n{'='*110}")
    print(f"  V7 REDUNDANT MULTI-VIEW RETRIEVAL BENCHMARK SUMMARY")
    print(f"{'='*110}")
    header1 = f"{'Variant':<28s} {'Top-1':>7s} {'Top-5':>7s} {'Top-10':>7s} {'Top-20':>7s} {'<=1px':>7s} {'<=3px':>7s} {'<=5px':>7s} {'<=10px':>7s}"
    print(header1)
    print("-" * len(header1))

    for var, st in results.items():
        print(f"{var:<28s} {st['Top1_Recall']:>6.1f}% {st['Top5_Recall']:>6.1f}% {st['Top10_Recall']:>6.1f}% {st['Top20_Recall']:>6.1f}% "
              f"{st['Acc_le1px']:>6.1f}% {st['Acc_le3px']:>6.1f}% {st['Acc_le5px']:>6.1f}% {st['Acc_le10px']:>6.1f}%")

    print(f"\n{'='*110}")
    header2 = f"{'Variant':<28s} {'MeanErr':>8s} {'MedErr':>8s} {'P95Err':>8s} {'MeanLat':>9s} {'P95Lat':>9s} {'PACE Activations':>18s}"
    print(header2)
    print("-" * len(header2))
    for var, st in results.items():
        print(f"{var:<28s} {st['Mean_Err']:>8.2f} {st['Median_Err']:>8.2f} {st['P95_Err']:>8.2f} {st['Mean_Lat']:>8.2f}ms {st['P95_Lat']:>8.2f}ms {st['pace_activations']:>14d}/{n}")

    # Write CSV Artifacts
    rows_ablation = []
    for var, st in results.items():
        rows_ablation.append({
            "Variant": var,
            "Top1_Recall": st["Top1_Recall"],
            "Top5_Recall": st["Top5_Recall"],
            "Top10_Recall": st["Top10_Recall"],
            "Top20_Recall": st["Top20_Recall"],
            "Acc_le1px": st["Acc_le1px"],
            "Acc_le3px": st["Acc_le3px"],
            "Acc_le5px": st["Acc_le5px"],
            "Acc_le10px": st["Acc_le10px"],
            "Mean_Err": st["Mean_Err"],
            "Median_Err": st["Median_Err"],
            "P95_Err": st["P95_Err"],
            "Mean_Lat_ms": st["Mean_Lat"],
            "P95_Lat_ms": st["P95_Lat"],
        })

    out_dir = os.path.join(ROOT_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    df_ablation = pd.DataFrame(rows_ablation)
    df_ablation.to_csv(os.path.join(out_dir, "v7_ablation.csv"), index=False)
    df_ablation[["Variant", "Top1_Recall", "Top5_Recall", "Top10_Recall", "Top20_Recall"]].to_csv(os.path.join(out_dir, "v7_candidate_recall.csv"), index=False)

    print(f"\nSaved 'results/v7_ablation.csv' and 'results/v7_candidate_recall.csv'.")
    return results


if __name__ == "__main__":
    evaluate_v7_benchmark()
