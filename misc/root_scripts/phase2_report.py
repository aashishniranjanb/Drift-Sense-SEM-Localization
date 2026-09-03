"""
phase2_report.py — produces every NUMBER and CHART the Phase 2 deck still needs.

INPUT: one CSV, dev_results.csv, with these columns (one row per development pair):

    pair_id      any id
    set          "A" | "B" | "C"          (C = absent pairs)
    present_gt   1 if the reference is really in the search image, else 0
    x_gt,y_gt    ground-truth centre (only meaningful when present_gt==1)
    theta_gt     ground-truth rotation, degrees CCW
    scale_gt     ground-truth down-scaling factor
    x_pred,y_pred,theta_pred,scale_pred   what register.py wrote
    found_pred   1/0
    score_pred   your confidence column
    runtime_s    wall-clock seconds for that pair

Everything the deck reports is derived from this one file, so there is exactly
one place where a number can be wrong.

USAGE:  python phase2_report.py dev_results.csv
OUTPUT: phase2_numbers.json + five PNGs in ./figures/
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- deck palette
INK, MUTED, BLUE, LIGHT = "#111827", "#5C6779", "#0B3D91", "#C7D3E8"
GREEN, AMBER, RED, GRID = "#1E7A44", "#9A6410", "#B3261E", "#EEF1F6"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": "#D7DCE5", "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
})
FIG = Path("figures")
FIG.mkdir(exist_ok=True)


# ------------------------------------------------- official Phase 2 credit rules
def loc_credit(err_px):
    if err_px <= 1: return 1.00
    if err_px <= 2: return 0.80
    if err_px <= 3: return 0.60
    if err_px <= 5: return 0.40
    return 0.00

def scale_credit(rel_err):          # |s_hat - s| / s
    if rel_err <= 0.01: return 1.00
    if rel_err <= 0.02: return 0.60
    if rel_err <= 0.05: return 0.30
    return 0.00

def rot_credit(abs_deg):            # |theta_hat - theta|
    if abs_deg <= 0.25: return 1.00
    if abs_deg <= 0.50: return 0.60
    if abs_deg <= 1.00: return 0.30
    return 0.00


def f1(tp, fp, fn):
    if tp == 0: return 0.0
    p, r = tp / (tp + fp), tp / (tp + fn)
    return 2 * p * r / (p + r)


def roc_auc(y_true, score):
    """AUC with no sklearn dependency (rank / Mann-Whitney formulation)."""
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(score, dtype=float)
    pos, neg = y.sum(), len(y) - y.sum()
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks over ties
    df = pd.DataFrame({"s": s, "r": ranks})
    ranks = df.groupby("s")["r"].transform("mean").to_numpy()
    return (ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)


def main(path):
    d = pd.read_csv(path)
    d["err_px"] = np.hypot(d.x_pred - d.x_gt, d.y_pred - d.y_gt)
    d.loc[d.present_gt == 0, "err_px"] = np.nan

    present = d[d.present_gt == 1].copy()
    present["credit"] = present.err_px.apply(loc_credit)
    # a pair we rejected earns no localization credit
    present.loc[present.found_pred == 0, "credit"] = 0.0

    A = present[present.set == "A"]
    B = present[present.set == "B"]
    credit_A, credit_B = A.credit.mean(), B.credit.mean()
    loc_points = (0.45 * credit_A + 0.55 * credit_B) * 40

    # ---- pose: scored ONLY where localization credit > 0
    scored = present[present.credit > 0].copy()
    scored["scale_rel"] = (scored.scale_pred - scored.scale_gt).abs() / scored.scale_gt
    scored["rot_abs"] = (scored.theta_pred - scored.theta_gt).abs()
    scored["sc_credit"] = scored.scale_rel.apply(scale_credit)
    scored["rt_credit"] = scored.rot_abs.apply(rot_credit)
    # mean over the SAME denominator localization used, per set, then 0.45/0.55
    def pose_component(col):
        a = scored[scored.set == "A"][col].sum() / max(len(A), 1)
        b = scored[scored.set == "B"][col].sum() / max(len(B), 1)
        return (0.45 * a + 0.55 * b) * 10
    scale_points, rot_points = pose_component("sc_credit"), pose_component("rt_credit")

    # ---- rejection F1, both conventions (report the ABSENT-as-positive one)
    gray = d[d.set.isin(["A", "B", "C"])]
    tp_p = int(((gray.present_gt == 1) & (gray.found_pred == 1)).sum())
    fn_p = int(((gray.present_gt == 1) & (gray.found_pred == 0)).sum())
    fp_p = int(((gray.present_gt == 0) & (gray.found_pred == 1)).sum())
    tn_p = int(((gray.present_gt == 0) & (gray.found_pred == 0)).sum())
    f1_present = f1(tp_p, fp_p, fn_p)
    f1_absent = f1(tn_p, fn_p, fp_p)          # absent treated as the positive class

    # ---- calibration AUC: score vs per-pair correctness
    gray = gray.copy()
    gray["correct"] = np.where(
        gray.present_gt == 1,
        ((gray.found_pred == 1) & (gray.err_px <= 5)).astype(int),
        (gray.found_pred == 0).astype(int),
    )
    auc = roc_auc(gray.correct, gray.score_pred)

    med_rt = float(gray.runtime_s.median())
    max_rt = float(gray.runtime_s.max())

    out = {
        "localization": {
            "set_A_mean_credit": round(float(credit_A), 4),
            "set_B_mean_credit": round(float(credit_B), 4),
            "points_of_40": round(float(loc_points), 2),
            "tier_counts_A": {k: int(v) for k, v in
                              A.err_px.apply(loc_credit).value_counts().items()},
            "tier_counts_B": {k: int(v) for k, v in
                              B.err_px.apply(loc_credit).value_counts().items()},
            "median_err_px_A": round(float(A.err_px.median()), 3),
            "median_err_px_B": round(float(B.err_px.median()), 3),
        },
        "pose": {
            "scale_points_of_10": round(float(scale_points), 2),
            "rotation_points_of_10": round(float(rot_points), 2),
            "total_of_20": round(float(scale_points + rot_points), 2),
            "scale_MAE_rel": round(float(scored.scale_rel.mean()), 5),
            "rotation_MAE_deg": round(float(scored.rot_abs.mean()), 4),
        },
        "rejection": {
            "confusion": {"present_found": tp_p, "present_rejected": fn_p,
                          "absent_found": fp_p, "absent_rejected": tn_p},
            "F1_present_as_positive": round(f1_present, 4),
            "F1_absent_as_positive": round(f1_absent, 4),
            "points_of_15": round(15 * f1_absent, 2),
        },
        "calibration": {"AUC": round(float(auc), 4), "points_of_10": round(10 * float(auc), 2)},
        "efficiency": {"median_runtime_s": round(med_rt, 3), "max_runtime_s": round(max_rt, 3),
                       "within_5s_median": bool(med_rt <= 5), "any_pair_over_20s": bool(max_rt > 20)},
    }
    out["base_total_estimate"] = round(
        loc_points + scale_points + rot_points + 15 * f1_absent + 10 * float(auc) + 5 + 10, 2)

    Path("phase2_numbers.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    # ============================================================ FIGURES
    # 1 — tier distribution, Set A vs Set B  → Results slide
    tiers = ["\u22641 px", "\u22642 px", "\u22643 px", "\u22645 px", ">5 px"]
    keys = [1.0, 0.8, 0.6, 0.4, 0.0]
    def pct(df):
        c = df.err_px.apply(loc_credit)
        return [100 * (c == k).mean() for k in keys]
    fig, ax = plt.subplots(figsize=(6.4, 2.9), dpi=200)
    x = np.arange(len(tiers)); w = 0.38
    ax.bar(x - w/2, pct(A), w, label="Set A — nominal", color=BLUE)
    ax.bar(x + w/2, pct(B), w, label="Set B — degraded", color=LIGHT)
    for i, (a, b) in enumerate(zip(pct(A), pct(B))):
        ax.text(i - w/2, a + 1, f"{a:.0f}%", ha="center", fontsize=7.5, color=INK)
        ax.text(i + w/2, b + 1, f"{b:.0f}%", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(x, tiers); ax.set_ylabel("% of set")
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(FIG / "p2_tier_distribution.png"); plt.close(fig)

    # 2 — localization error CDF with the four credit thresholds
    fig, ax = plt.subplots(figsize=(6.4, 2.9), dpi=200)
    for df, lab, col in [(A, "Set A", BLUE), (B, "Set B", AMBER)]:
        e = np.sort(df.err_px.dropna().to_numpy())
        ax.step(e, 100 * np.arange(1, len(e) + 1) / len(e), where="post", color=col, lw=1.8, label=lab)
    for t in (1, 2, 3, 5):
        ax.axvline(t, color=MUTED, lw=0.7, ls=":")
        ax.text(t, 2, f"{t}px", fontsize=7, color=MUTED, ha="center")
    ax.set_xscale("log"); ax.set_xlabel("localization error (px, log scale)")
    ax.set_ylabel("cumulative % of pairs"); ax.set_ylim(0, 100)
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "p2_error_cdf.png"); plt.close(fig)

    # 3 — calibration ROC  → proves the score column earns its 10 points
    ys = gray.sort_values("score_pred", ascending=False)
    tpr = np.cumsum(ys.correct) / max(ys.correct.sum(), 1)
    fpr = np.cumsum(1 - ys.correct) / max((1 - ys.correct).sum(), 1)
    fig, ax = plt.subplots(figsize=(3.4, 3.0), dpi=200)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=":")
    ax.plot(np.r_[0, fpr], np.r_[0, tpr], color=BLUE, lw=2)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title(f"AUC = {auc:.4f}", color=INK, fontsize=10, loc="left")
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(FIG / "p2_calibration_roc.png"); plt.close(fig)

    # 4 — runtime histogram against the 5 s median budget  → Feasibility slide
    fig, ax = plt.subplots(figsize=(6.0, 2.5), dpi=200)
    ax.hist(gray.runtime_s, bins=30, color=BLUE)
    ax.axvline(med_rt, color=GREEN, lw=1.6, label=f"median {med_rt:.2f} s")
    ax.axvline(5, color=AMBER, lw=1.2, ls="--", label="5 s median budget")
    ax.set_xlabel("wall-clock seconds per pair"); ax.set_ylabel("pairs")
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "p2_runtime_hist.png"); plt.close(fig)

    # 5 — pose accuracy against the credit bands
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), dpi=200)
    axes[0].hist(100 * scored.scale_rel, bins=30, color=BLUE)
    for t, c in [(1, GREEN), (2, AMBER), (5, RED)]:
        axes[0].axvline(t, color=c, lw=1.1, ls="--")
    axes[0].set_xlabel("scale error (%)  bands 1 / 2 / 5"); axes[0].set_ylabel("pairs")
    axes[1].hist(scored.rot_abs, bins=30, color=BLUE)
    for t, c in [(0.25, GREEN), (0.5, AMBER), (1.0, RED)]:
        axes[1].axvline(t, color=c, lw=1.1, ls="--")
    axes[1].set_xlabel("rotation error (deg)  bands 0.25 / 0.5 / 1.0")
    for a in axes:
        a.yaxis.grid(True, color=GRID, lw=0.8); a.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(FIG / "p2_pose_accuracy.png"); plt.close(fig)

    print("\nWrote phase2_numbers.json and 5 PNGs to ./figures/")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev_results.csv")
