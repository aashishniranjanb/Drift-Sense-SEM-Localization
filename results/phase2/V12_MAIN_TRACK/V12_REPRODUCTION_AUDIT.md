# V12 Reproduction Audit

This document traces and resolves the apparent discrepancy in V12 candidate retrieval recall measurements (51.43% vs. 60.71% / 61.43% Top-100).

## 1. Reproduction Setup & Hash Verification
- **Dataset**: `data/phase2_dev/pairs.csv`
- **Row Count**: 180 total pairs (70 Set A, 70 Set B, 40 Set C)
- **Present-Case Target Count**: 140 pairs (Set A + Set B)
- **Active Git Commit**: `1d28456`
- **Evaluation Script**: `phase2/v12_candidate_recovery_study.py`

---

## 2. Quantitative Verification & Resolution

The discrepancy is the direct mathematical consequence of the **Candidate Pool Dilution Effect** when merging multiple hypotheses:

### A. Configuration 1: Single-Pose Hypothesis + Raw NMS Sweep
*   **Method**: Run scale & rotation search once ($H=1$). Extract candidates directly from this single correlation plane.
*   **Result (NMS $r=5$)**: Top-100 Recall = **60.71%** (85/140 present cases).
*   **Result (NMS $r=10$)**: Top-50 Recall = **51.43%** (72/140 present cases).
*   *Mechanism*: Sharp peaks are preserved. The pool is not flooded with noise from other candidate poses.

### B. Configuration 2: Naive Multi-Pose + Multi-Channel Union (The Dilution Failure)
*   **Method**: Run $H=3$ scale/rotation hypotheses. For each hypothesis, extract 6 candidate lists (LocalMax $w=2$, $w=4$, Percentile 99, NMS $r=5$, Gradient Maxima, Gradient Percentile). Merge them into a single flat list and take the Top-100 by un-normalized correlation score.
*   **Result**: Top-100 Recall = **51.43%** (72/140 present cases).
*   *Mechanism*: False peaks from the 2 sub-optimal pose templates (which are slightly blurred due to rotation interpolation) have high un-normalized correlation scores. They crowd and flood the Top-100 slots, pushing the true peak of the correct pose down below the 100-candidate cap (`RESCUE_DENSITY_CAP` = 32.86%).

---

## 3. Verified Base Control Numbers

The reproduced control numbers for the V12 base candidates under $H=1$:
*   **Top-20 Recall**: **40.00%** (NMS $r=5$) / **40.71%** (NMS $r=10$)
*   **Top-50 Recall**: **50.00%** (NMS $r=5$) / **51.43%** (NMS $r=10$)
*   **Top-100 Recall**: **60.71%** (NMS $r=5$) / **59.29%** (NMS $r=10$)
*   **Oracle Ceiling**: **63.57%**
*   **Latency**: **17.6 ms** (extraction stage) / **2.95s** (full pipeline)
