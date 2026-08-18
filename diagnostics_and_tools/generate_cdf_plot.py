"""
Generate Cumulative Distribution Function (CDF) Curve Plot and Multi-Anchor Visualization
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd

from anchor_consensus import select_distinctive_anchors, normalize_intensity

def generate_cdf_figure():
    csv_path = "results/cdf_error_distribution.csv"
    if not os.path.exists(csv_path):
        return

    df_cdf = pd.read_csv(csv_path, index_col=0)
    fig, ax = plt.subplots(figsize=(8, 5))

    thresholds = [1, 2, 3, 4, 5, 10, 25, 50, 100]

    for col in df_cdf.columns:
        vals = df_cdf[col].iloc[:len(thresholds)].values
        ax.plot(thresholds, vals, marker='o', label=col, linewidth=2)

    ax.set_xscale('log')
    ax.set_xlabel("Localization Error Tolerance (pixels) [Log Scale]", fontsize=11)
    ax.set_ylabel("Cumulative Accuracy P(error <= e) %", fontsize=11)
    ax.set_title("Localization Error Cumulative Distribution Function (CDF)", fontsize=13, fontweight='bold')
    ax.set_xticks(thresholds)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(0, 105)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(fontsize=10)

    plt.tight_layout()
    out_path = "results/error_cdf_curve.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Wrote CDF plot to '{out_path}'")


def generate_anchor_illustration():
    manifest_path = "data/benchmark_120/manifest.csv"
    if not os.path.exists(manifest_path):
        return

    df_manifest = pd.read_csv(manifest_path)
    sample_indices = [0, 10, 45, 75]

    for idx in sample_indices:
        if idx >= len(df_manifest):
            continue
        row = df_manifest.iloc[idx]
        ref_img = cv2.imread(row["reference_path"], cv2.IMREAD_GRAYSCALE)
        arch = row["architecture"]
        diff = row["difficulty"]

        ref_100 = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)
        anchors = select_distinctive_anchors(ref_100, patch_size=36, stride=12, num_anchors=3)

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(ref_img, cmap="gray")
        axs[0].set_title(f"Native Reference (1000x1000)\n[{arch} - {diff}]")
        axs[0].axis("off")

        ref_rgb = cv2.cvtColor(ref_100, cv2.COLOR_GRAY2RGB)
        colors = [(255, 0, 0), (0, 255, 0), (0, 150, 255)]

        for i, a in enumerate(anchors):
            x, y, ps = a["x"], a["y"], a["patch_size"]
            c = colors[i % len(colors)]
            cv2.rectangle(ref_rgb, (x, y), (x+ps, y+ps), c, 2)
            cv2.putText(ref_rgb, f"A{i+1}", (x+3, y+12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)

        axs[1].imshow(ref_rgb)
        axs[1].set_title("100x100 Template with\nDistinctive Anchors Selected")
        axs[1].axis("off")

        out_path = f"results/diagnostics/anchor_demo_{idx:03d}_{arch}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved anchor demo: '{out_path}'")


if __name__ == "__main__":
    generate_cdf_figure()
    generate_anchor_illustration()
