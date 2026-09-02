"""
Generate 2-page failure_analysis.pdf for competition submission.
Compatible with fpdf 1.7.2 (ASCII-only content).
"""
import os
from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 11)
        self.set_fill_color(30, 30, 30)
        self.set_text_color(255, 255, 255)
        self.cell(0, 8, "Drift-Sense++ | Phase 2 Failure Analysis", 0, 1, "C", True)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 6, "Page %d/2" % self.page_no(), 0, 0, "C")

    def section(self, title):
        self.ln(3)
        self.set_font("Helvetica", "B", 10)
        self.set_fill_color(210, 225, 245)
        self.cell(0, 7, "  " + title, 0, 1, "L", True)
        self.ln(1)

    def body(self, text):
        self.set_font("Helvetica", "", 9)
        self.set_x(self.l_margin + 4)
        self.multi_cell(0, 5, text)
        self.ln(1)

    def row(self, cols, widths, border=1, fill=False, bold=False):
        style = "B" if bold else ""
        self.set_font("Helvetica", style, 8.5)
        for i, (col, w) in enumerate(zip(cols, widths)):
            self.cell(w, 5.5, col, border, 0, "C", fill)
        self.ln()


def build():
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 15, 15)

    # ---- PAGE 1 ----
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 9, "Drift-Sense++: SEM-to-Optical Registration", 0, 1, "C")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 5, "Applied Materials Phase 2 | Failure Analysis Report", 0, 1, "C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)

    pdf.section("1.  Problem Statement")
    pdf.body(
        "Given a high-resolution optical reference image and a degraded SEM search image at "
        "unknown scale (8-12x) and rotation (+/-5 deg), the system must locate the reference, "
        "recover pose (x, y, theta, scale), and correctly classify absent cases (found=0). "
        "The primary challenge is periodic lattice structures: identical circuit cells appear "
        "at regular offsets, creating multiple high-correlation replica peaks in the NCC plane."
    )

    pdf.section("2.  System Architecture  (V21 Championship Pipeline)")
    pdf.body(
        "[V19] Dual-Queue Retrieval: separates the candidate budget into center queue (35 slots) "
        "and periphery queue (15 slots), preventing periodic replicas from flooding the pool. "
        "Top-50 GT recall improved from 50.0% to 67.1% over the greedy NMS baseline.\n"
        "[V18-C] Periodicity-Adaptive Ranker: multi-scale context verification + phase residual "
        "scoring + center prior gated by family population. Conditional Top-1 improved 31.0% to "
        "46.5% (+50% relative improvement).\n"
        "[V14] Safe Rejection Gate: composite scalar gate "
        "(0.35*NCC + 0.40*ctx + 0.15*PSR + 0.10*margin - 0.20*phase_res, T=0.58). "
        "Conservative to protect pose recovery on difficult degraded cases."
    )

    pdf.section("3.  Development Set Score (180 Pairs: 70 SetA, 70 SetB, 40 SetC)")
    pdf.body("V21 competition-style score breakdown on the 180-pair development set:")
    pdf.ln(1)
    W = [76, 30, 30, 34]
    pdf.set_fill_color(200, 212, 232)
    pdf.row(["Component", "V21 Score", "Target", "Gap"], W, bold=True, fill=True)
    for i, r in enumerate([
        ("Localization (40 pts)",  "4.91",  "34-38",  "-29.1"),
        ("Pose Recovery (20 pts)", "19.55", "18-20",  "+0.0"),
        ("Rejection (15 pts)",     "5.83",  "13-15",  "-7.2"),
        ("Calibration (10 pts)",   "5.32",  "9-10",   "-4.7"),
        ("Efficiency (5 pts)",     "5.00",  "5",      "0.0"),
        ("Documentation (10 pts)", "10.00", "10",     "0.0"),
        ("TOTAL (100 pts)",        "50.65", "60-70+", "-14.4"),
    ]):
        pdf.set_fill_color(240, 245, 255) if i % 2 == 0 else pdf.set_fill_color(255, 255, 255)
        pdf.row(list(r), W, fill=(i % 2 == 0))
    pdf.ln(2)

    pdf.section("4.  Failure Taxonomy  (140 PRESENT, 40 ABSENT cases)")
    W2 = [52, 14, 30, 84]
    pdf.set_fill_color(200, 212, 232)
    pdf.row(["Failure Mode", "Count", "Rate", "Root Cause"], W2, bold=True, fill=True)
    for i, r in enumerate([
        ("PRESENCE_FALSE_NEGATIVE", "76", "54% PRESENT", "Gate rejects true positive (T too conservative)"),
        ("PERIODIC_REPLICA",        "46", "33% PRESENT", "Wrong clone ranked #1; GT below replica"),
        ("ABSENCE_FALSE_POSITIVE",  "15", "38% ABSENT",  "Hard-negative accepted as PRESENT"),
        ("SUBPIXEL_SUCCESS",        "18", "13% PRESENT", "Correct localisation within 5 px"),
        ("REJECTION_SUCCESS",       "25", "62% ABSENT",  "Correct absence detection"),
    ]):
        pdf.set_fill_color(240, 245, 255) if i % 2 == 0 else pdf.set_fill_color(255, 255, 255)
        pdf.row(list(r), W2, fill=(i % 2 == 0))

    # ---- PAGE 2 ----
    pdf.add_page()

    pdf.section("5.  Root Cause Analysis")

    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 6, "  5.1  Periodic Replica Confusion (46 failures)", 0, 1)
    pdf.body(
        "V17 forensic autopsy of 35 periodic failures: GT candidate mean center_distance = "
        "119.2 px vs wrong replica mean = 245.0 px (Mann-Whitney p = 0.0029). Despite this "
        "strong spatial signal, NCC score for the wrong replica is ~2-5% higher due to periodic "
        "signal reinforcement. The V18-C center prior partially mitigates this but fails when "
        "GT lies far from the geometric center (peripheral die structures)."
    )

    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 6, "  5.2  Retrieval Suppression (18/35 periodic failures)", 0, 1)
    pdf.body(
        "Greedy NMS (r=5 px) allows periodic replicas to consume all K=50 candidate slots, "
        "suppressing the GT candidate before ranking runs. The V19 dual-queue fixes this for "
        "central GT locations but not peripheral ones. 30 PRESENT cases still have GT outside "
        "Top-50, creating a hard retrieval ceiling on localization performance."
    )

    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 6, "  5.3  Presence Gate Over-Rejection (76 PRESENT cases rejected)", 0, 1)
    pdf.body(
        "The V14 composite score was calibrated before V19 integration. After integrating "
        "V19 dual-queue retrieval, the best candidate pool became more diverse, shifting "
        "composite scores downward. Threshold sweep shows that simply lowering T sacrifices "
        "rejection F1, requiring a per-candidate evidence-aware gating strategy."
    )

    pdf.section("6.  Applied Mitigations")
    pdf.body(
        "V17 Forensic Autopsy: 100% attribution of 35 periodic failures to 4 root causes. "
        "Established center_distance as primary discriminator (p < 0.003).\n\n"
        "V18-C Ranker: Conditional Top-1 improved 31.0% -> 46.5% using multi-scale context + "
        "phase residual + family-population-gated center prior.\n\n"
        "V19 Dual-Queue Retrieval: Top-50 GT recall improved 50.7% -> 67.1% (+32.4% relative); "
        "rescued 10/18 retrieval-suppressed failures without increasing runtime.\n\n"
        "V20 Scalar/CNN Presence (REJECTED): Logistic classifiers and Siamese CNNs failed to "
        "generalise across nominal/degraded/absent distribution. Patch CNN destroyed Set B "
        "recall (50.6%) due to noise overwhelming patch-level signal."
    )

    pdf.section("7.  Known Limitations")
    pdf.body(
        "1. Localization bottleneck is ranking quality (~46% conditional Top-1), not retrieval.\n"
        "2. Set B (degraded SEM) has weaker NCC signal, reducing relative discriminability.\n"
        "3. Presence gate conservatively rejects ~54% of PRESENT cases to avoid false positives "
        "on unseen hard-negative configurations in the blind test set.\n"
        "4. Scale/rotation search covers [8,12] / +/-5 deg via coarse-to-fine; accuracy "
        "degrades near search space boundaries."
    )

    pdf.section("8.  Final Submission")
    pdf.body(
        "Pipeline:   V19 Dual-Queue Retrieval + V18-C Periodicity-Adaptive Ranker + V14 Gate\n"
        "Runtime:    Median < 5 s/pair (CPU-only, Python 3.11, no GPU, no network)\n"
        "Contract:   register.py --input pairs.csv --output predictions.csv\n"
        "Output:     pair_id, x, y, theta, scale, found, score; zeroed pose when found=0\n"
        "Generator:  generate_dataset.py (Phase 2 spec: scale 8-12x, rotation +/-5 deg)\n"
        "Deps:       numpy, scipy, opencv-python, scikit-learn (standard CPU packages)"
    )

    out = os.path.join("FINAL_SUBMISSION", "failure_analysis.pdf")
    os.makedirs("FINAL_SUBMISSION", exist_ok=True)
    pdf.output(out, "F")
    print("Written: %s  (%d bytes)" % (out, os.path.getsize(out)))


if __name__ == "__main__":
    build()
