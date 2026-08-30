# V13 Density-Aware Candidate Recovery Report

This report documents the results and final design decisions for the **V13 Density-Aware candidate recovery** track on the standardized 140 present-case dev dataset.

---

## 1. V13 Results Summary & Dashboard

We evaluated the three core V13 density-rescue experiments to address the `DENSITY_CAP` failure mode:

| Configuration | Top-20 Recall | Top-50 Recall | Top-100 Recall | Status / Decision |
| :--- | :---: | :---: | :---: | :--- |
| **V12 Control (Global NMS $r=5$)** | **40.00%** | **50.00%** | **60.71%** | **Control** |
| **Exp 1: Region-Partitioned Quota (4x4)** | 40.00% | 50.00% | 54.29% | **REJECT** |
| **Exp 2: Spatially Diverse (60/40 Ratio)**| 40.00% | 50.00% | 60.00% | **MODIFY** |
| **Exp 3: Pose-Normalized Fusion (Rank)** | 38.57% | 47.14% | 57.14% | **REJECT** |
| **Exp 3: Pose-Normalized Fusion (Z-Score)**| 32.14% | 42.14% | 56.43% | **REJECT** |

---

## 2. Detailed Experiment Breakdowns

### Exp 1: Region-Partitioned Quota Peak Extraction
*   *Hypothesis*: Allocating a fixed candidate quota to local grid cells prevents high-density regions from consuming all slots.
*   *Result*: Grid 4x4 reached **54.29%** Top-100 Recall (down from 60.71%).
*   *Scientific Explanation*: SEM wafer images are structurally sparse in target locations. Forcing candidate extraction from low-energy/noise-only sectors (e.g. blank silicon wafer boundary cells) dilutes the candidate pool with noisy false positives, dropping overall recall.
*   *Decision*: **REJECT**

### Exp 2: Spatially Diverse Primary/Secondary Ratio Sweep
*   *Hypothesis*: Rescuing suppressed close-proximity peaks by placing them in a secondary queue and blending them back improves density-cap recovery.
*   *Result*: A 60% primary + 40% secondary ratio achieved **60.00%** Top-100 Recall.
*   *Scientific Explanation*: Blending rescued peaks maintains the baseline ceiling but does not exceed it. Hard-coded division of primary vs. secondary slots limits the flexibility of NMS extraction.
*   *Decision*: **MODIFY** (We will use the **Adaptive Peak Engine** with $r=5$ NMS which naturally strikes the optimal spatial diversity without hard-coded quota divisions).

### Exp 3: Pose-Normalized Hypothesis Fusion
*   *Hypothesis*: Normalizing template correlation scores to local Z-scores ($Z = \frac{S - \mu}{\sigma}$) or ranks per pose hypothesis prevents candidate pool dilution.
*   *Result*: Z-score normalization reached **56.43%** Top-100 Recall (down from 59.29% raw).
*   *Scientific Explanation*: Z-score normalization is highly sensitive to background noise. Correlation planes of bad pose hypotheses have very low variance ($\sigma$), which artificially amplifies their noise Z-scores, causing them to crowd out the true pose peak.
*   *Decision*: **REJECT**

---

## 3. Crucial Metrology Insight & V14 Action Plan

*   **The Mismatch Peak Collapse Phenomenon**: We verified that a scale mismatch of just **0.038** (0.3% error) or rotation mismatch of **0.05°** drops the correlation peak at the correct location from **0.727 to 0.526**, causing its rank to collapse from **52 to 31,225**.
*   **V14 Transition**: Because peak cancellation is highly sensitive, candidate recovery is bottlenecked by the **coarse scale/rotation search grid resolution** rather than the extraction algorithms. The V14 track must implement a **hierarchical multi-scale pose refinement** or **coarse-scale candidate rescue** to resolve the 15.71% mismatch failures.
