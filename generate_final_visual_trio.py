"""
Generate Final Visual Trio for Submission Package:
  1. 03_confidence_gate.png (FFT -> High Confidence -> Trust vs Ambiguous -> PACE -> Phase -> Subpixel -> Output)
  2. 05_ablation_comparison.png (Scientific Progression V1 -> V7 Bar Chart: <=5px Acc, Median, Latency)
  3. 06_system_overview.png (End-to-End SAFE-CAR System Architecture Diagram)
Saves all visuals in submission_package/visuals/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def generate_03_confidence_gate():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.axis("off")
    ax.set_title("DRIFT-SENSE++ SAFE-CAR CONFIDENCE SAFETY GATE ARCHITECTURE", fontsize=13, fontweight="bold", pad=15)

    # Draw diagram boxes
    bbox_props = dict(boxstyle="round,pad=0.5", fc="lightskyblue", ec="b", lw=2)
    bbox_gate = dict(boxstyle="round,pad=0.6", fc="gold", ec="darkorange", lw=2)
    bbox_trust = dict(boxstyle="round,pad=0.5", fc="lightgreen", ec="g", lw=2)
    bbox_pace = dict(boxstyle="round,pad=0.5", fc="plum", ec="purple", lw=2)
    bbox_out = dict(boxstyle="round,pad=0.5", fc="palegreen", ec="darkgreen", lw=2)

    ax.text(0.5, 0.90, "FULL TEMPLATE DUAL-CHANNEL FFT\n(Intensity FFT + Scharr Gradient FFT)", ha="center", va="center", bbox=bbox_props, fontsize=10, fontweight="bold")
    ax.text(0.5, 0.68, "CONFIDENCE SAFETY GATE\n(Delta-S >= 0.010 & PSR >= 5.5?)", ha="center", va="center", bbox=bbox_gate, fontsize=10, fontweight="bold")

    # High Confidence Path (Left)
    ax.text(0.20, 0.45, "HIGH CONFIDENCE (62%)\nTrust Classical Peak\n(30 ms Fast Path)", ha="center", va="center", bbox=bbox_trust, fontsize=9, fontweight="bold")

    # Ambiguous Path (Right)
    ax.text(0.80, 0.45, "PERIODIC AMBIGUITY (38%)\nPACE Neural Residual Ranker\n(106k params, Group Ranking)", ha="center", va="center", bbox=bbox_pace, fontsize=9, fontweight="bold")

    ax.text(0.5, 0.25, "PHASE CORRELATION & 2D PARABOLOID FIT\nDual Estimator Subpixel Consensus (D <= 2.0 px)", ha="center", va="center", bbox=bbox_props, fontsize=10, fontweight="bold")

    ax.text(0.5, 0.08, "FINAL LOCALIZED COORDINATE\n(x.xx, y.yy)", ha="center", va="center", bbox=bbox_out, fontsize=11, fontweight="bold")

    # Arrows
    arrow_props = dict(arrowstyle="->", lw=2, color="navy")
    ax.annotate("", xy=(0.5, 0.76), xytext=(0.5, 0.83), arrowprops=arrow_props)
    ax.annotate("", xy=(0.20, 0.52), xytext=(0.40, 0.65), arrowprops=dict(arrowstyle="->", lw=2, color="green"))
    ax.annotate("", xy=(0.80, 0.52), xytext=(0.60, 0.65), arrowprops=dict(arrowstyle="->", lw=2, color="purple"))

    ax.annotate("", xy=(0.40, 0.29), xytext=(0.20, 0.38), arrowprops=dict(arrowstyle="->", lw=2, color="green"))
    ax.annotate("", xy=(0.60, 0.29), xytext=(0.80, 0.38), arrowprops=dict(arrowstyle="->", lw=2, color="purple"))
    ax.annotate("", xy=(0.5, 0.13), xytext=(0.5, 0.20), arrowprops=arrow_props)

    plt.tight_layout()
    out_p = "submission_package/visuals/03_confidence_gate.png"
    plt.savefig(out_p, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_p}")


def generate_05_ablation_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    fig.suptitle("SCIENTIFIC PROGRESSION & ABLATION COMPARISON ACROSS ALL 7 ITERATIONS", fontsize=12, fontweight="bold")

    iterations = ["V1 Baseline", "V2 MultiScale", "V3 Adaptive", "V4 HCR", "V5 PACE", "V6 SAFE-CAR*", "V7 MultiView"]
    acc_le5 = [66.0, 49.2, 44.2, 63.5, 64.5, 66.0, 55.5]
    latencies = [30.2, 1981.4, 418.4, 75.8, 44.6, 139.2, 877.1]

    colors = ["gray", "gray", "gray", "red", "orange", "darkgreen", "crimson"]

    # Accuracy Bar Chart
    bars1 = ax1.bar(iterations, acc_le5, color=colors, edgecolor="black")
    ax1.set_title("In-Bounds Accuracy (<=5px %)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Accuracy (%)", fontsize=10)
    ax1.set_ylim(0, 80)
    ax1.tick_params(axis="x", rotation=45)
    ax1.axhline(66.0, color="green", linestyle="--", alpha=0.7, label="V6 Baseline Ceiling (66%)")
    ax1.legend(loc="upper right", fontsize=8)

    for bar, val in zip(bars1, acc_le5):
        ax1.text(bar.get_x() + bar.get_width()/2.0, val + 1.5, f"{val:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Latency Bar Chart (Log Scale)
    bars2 = ax2.bar(iterations, latencies, color=colors, edgecolor="black")
    ax2.set_yscale("log")
    ax2.set_title("Mean End-to-End Latency (ms, Log Scale)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Latency (ms)", fontsize=10)
    ax2.tick_params(axis="x", rotation=45)

    for bar, val in zip(bars2, latencies):
        ax2.text(bar.get_x() + bar.get_width()/2.0, val * 1.15, f"{val:.1f}ms", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    out_p = "submission_package/visuals/05_ablation_comparison.png"
    plt.savefig(out_p, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_p}")


def generate_06_system_overview():
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ax.axis("off")
    ax.set_title("DRIFT-SENSE++ SAFE-CAR END-TO-END SYSTEM OVERVIEW", fontsize=13, fontweight="bold", pad=15)

    # Draw horizontal pipeline boxes
    boxes = [
        ("INPUT\nRef (1000x1000)\nSearch (1000x1000)", "lightblue", 0.08),
        ("DUAL FFT RETRIEVAL\nIntensity + Gradient\n(Top-20 Candidates)", "lightcyan", 0.28),
        ("CONFIDENCE GATE\nDelta-S & PSR Signal\nCheck", "gold", 0.48),
        ("SAFE AI RANKER\n(Activated ONLY under\nperiodic ambiguity)", "plum", 0.68),
        ("SUBPIXEL METROLOGY\nPhase Correlation +\nParaboloid Consensus", "palegreen", 0.88),
    ]

    for label, color, xpos in boxes:
        rect = patches.FancyBboxPatch((xpos - 0.08, 0.35), 0.16, 0.30, boxstyle="round,pad=0.03", fc=color, ec="black", lw=1.5)
        ax.add_patch(rect)
        ax.text(xpos, 0.50, label, ha="center", va="center", fontsize=9, fontweight="bold")

    # Arrows between stages
    for i in range(len(boxes) - 1):
        x_start = boxes[i][2] + 0.08
        x_end = boxes[i+1][2] - 0.08
        ax.annotate("", xy=(x_end, 0.50), xytext=(x_start, 0.50), arrowprops=dict(arrowstyle="->", lw=2, color="darkblue"))

    # Bottom Callout text
    ax.text(0.5, 0.10, "Winning Production Message: Fast Trusted Classical Path (30ms) on 62% of captures | AI activated ONLY when required",
            ha="center", va="center", fontsize=10, fontweight="bold", color="darkgreen", bbox=dict(boxstyle="square,pad=0.5", fc="honeydew", ec="green", lw=1.5))

    plt.tight_layout()
    out_p = "submission_package/visuals/06_system_overview.png"
    plt.savefig(out_p, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_p}")


def main():
    os.makedirs("submission_package/visuals", exist_ok=True)
    generate_03_confidence_gate()
    generate_05_ablation_comparison()
    generate_06_system_overview()


if __name__ == "__main__":
    main()
