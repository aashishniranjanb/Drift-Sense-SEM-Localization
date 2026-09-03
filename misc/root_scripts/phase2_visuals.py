"""
phase2_visuals.py — the three qualitative panels the deck has dashed boxes for.

Produces:
  figures/p2_success_panel.png     Reference | Search + prediction vs GT | zoom
  figures/p2_failure_panel.png     same layout, on a periodic-replica failure
  figures/p2_rgb_panel.png         RGB reference | RGB search | localisation result

Call it from your own benchmark loop, or edit the __main__ block with three
pair ids you already know are (a) a clean win, (b) a lattice-shift failure.
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

INK, MUTED, BLUE = "#111827", "#5C6779", "#0B3D91"
GREEN, RED = "#1E7A44", "#B3261E"
FIG = Path("figures"); FIG.mkdir(exist_ok=True)


def _load(p):
    from PIL import Image
    return np.asarray(Image.open(p))


def localization_panel(ref_path, search_path, pred_xy, gt_xy, scale, theta,
                       out_name, verdict, note):
    """verdict: 'SUCCESS' or 'FAILURE'. pred_xy / gt_xy are (x, y) in search coords."""
    ref, search = _load(ref_path), _load(search_path)
    err = float(np.hypot(pred_xy[0] - gt_xy[0], pred_xy[1] - gt_xy[1]))
    box = ref.shape[0] / scale                       # footprint of the reference

    fig, ax = plt.subplots(1, 3, figsize=(9.0, 3.2), dpi=200)

    ax[0].imshow(ref, cmap="gray"); ax[0].set_title("REFERENCE  1000×1000", fontsize=8,
                                                    color=BLUE, loc="left")

    ax[1].imshow(search, cmap="gray")
    ax[1].add_patch(Rectangle((gt_xy[0] - box/2, gt_xy[1] - box/2), box, box,
                              ec=GREEN, fc="none", lw=1.6, label="ground truth"))
    ax[1].add_patch(Rectangle((pred_xy[0] - box/2, pred_xy[1] - box/2), box, box,
                              ec=RED if verdict == "FAILURE" else BLUE,
                              fc="none", lw=1.6, ls="--", label="prediction"))
    ax[1].set_title(f"SEARCH  ·  scale {scale:.2f}  ·  θ {theta:+.2f}°",
                    fontsize=8, color=BLUE, loc="left")
    ax[1].legend(fontsize=7, frameon=False, loc="lower right", labelcolor=INK)

    # zoom on the ground-truth neighbourhood so both markers are visible
    pad = max(box * 1.6, err * 1.4, 40)
    cx, cy = (gt_xy[0] + pred_xy[0]) / 2, (gt_xy[1] + pred_xy[1]) / 2
    ax[2].imshow(search, cmap="gray")
    ax[2].set_xlim(cx - pad, cx + pad); ax[2].set_ylim(cy + pad, cy - pad)
    ax[2].add_patch(Circle(gt_xy, pad*0.05, ec=GREEN, fc="none", lw=1.8))
    ax[2].add_patch(Circle(pred_xy, pad*0.05, ec=RED if verdict == "FAILURE" else BLUE,
                           fc="none", lw=1.8))
    ax[2].plot([gt_xy[0], pred_xy[0]], [gt_xy[1], pred_xy[1]], color=MUTED, lw=1, ls=":")
    ax[2].set_title(f"ZOOM  ·  error {err:.2f} px", fontsize=8,
                    color=RED if verdict == "FAILURE" else GREEN, loc="left")

    for a in ax: a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{verdict} — {note}", fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG / out_name); plt.close(fig)
    print(f"wrote {out_name}  (error {err:.2f} px)")


def rgb_panel(ref_rgb, search_rgb, pred_xy, gt_xy, scale, out_name="p2_rgb_panel.png"):
    ref, search = _load(ref_rgb), _load(search_rgb)
    box = ref.shape[0] / scale
    err = float(np.hypot(pred_xy[0] - gt_xy[0], pred_xy[1] - gt_xy[1]))
    fig, ax = plt.subplots(1, 3, figsize=(9.0, 3.2), dpi=200)
    ax[0].imshow(ref); ax[0].set_title("RGB REFERENCE", fontsize=8, color=BLUE, loc="left")
    ax[1].imshow(search); ax[1].set_title("RGB WIDE SEARCH", fontsize=8, color=BLUE, loc="left")
    ax[2].imshow(search)
    ax[2].add_patch(Rectangle((gt_xy[0]-box/2, gt_xy[1]-box/2), box, box, ec=GREEN, fc="none", lw=1.6))
    ax[2].add_patch(Rectangle((pred_xy[0]-box/2, pred_xy[1]-box/2), box, box, ec=BLUE,
                              fc="none", lw=1.6, ls="--"))
    ax[2].set_title(f"LOCALISATION  ·  error {err:.2f} px", fontsize=8, color=GREEN, loc="left")
    for a in ax: a.set_xticks([]); a.set_yticks([])
    fig.suptitle("BONUS EXTENSION — RGB OPTICAL PATH (Set D analogue)",
                 fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG / out_name); plt.close(fig)
    print(f"wrote {out_name}")


if __name__ == "__main__":
    # Clean win from Set A / Set B
    localization_panel(
        ref_path="data/phase2_dev/reference/pair_033.png",
        search_path="data/phase2_dev/search/pair_033.png",
        pred_xy=(506.53, 415.50), gt_xy=(506.49, 415.49),
        scale=9.00, theta=1.75,
        out_name="p2_success_panel.png", verdict="SUCCESS",
        note="distinctive local structure; classical evidence accepted by the gate",
    )
    # Set B lattice-shift failure (global-argmax vs nearest-to-centre)
    localization_panel(
        ref_path="data/phase2_dev/reference/pair_070.png",
        search_path="data/phase2_dev/search/pair_070.png",
        pred_xy=(529.59, 557.58), gt_xy=(534.00, 486.68),
        scale=10.70, theta=-3.75,
        out_name="p2_failure_panel.png", verdict="FAILURE",
        note="periodic replica one lattice pitch away in Set B; global-argmax differs from nearest-to-centre",
    )
    # RGB Optical Path
    rgb_panel("SlideImagesOnly/reference_rgb.png", "SlideImagesOnly/search_rgb.png",
              pred_xy=(511.0, 402.0), gt_xy=(511.0, 402.0), scale=10.0)

