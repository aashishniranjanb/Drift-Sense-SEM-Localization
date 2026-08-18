# Drift-Sense Proposed Architecture & Baseline Benchmark Evaluation Report

## 1. Executive Summary

This report documents the diagnostic investigation and implementation of the **Robust Multi-Scale Structural Registration & Verification Architecture** proposed in [PROPOSED_SOLUTION.md](../PROPOSED_SOLUTION.md) for the **Applied Materials Drift-Sense AI** challenge.

### Key Milestones & Breakthroughs:
1. **Diagnostic Isolation (Retrieval vs. Ranking)**:
   - **Clean-Case Oracle Validation** confirmed that the generator ground-truth coordinate math is subpixel-accurate ($0.10 - 0.36$ px error on clean non-ambiguous samples), with 100% of errors occurring as exact periodic lattice shifts.
   - **Top-K Recall Analysis** uncovered that plain template retrieval achieves **60.83% Top-10 recall** across the 120-case dataset, proving that ranking and periodic replica ambiguity was the primary bottleneck holding baseline Top-1 accuracy at **35.83%**.
2. **Proposed Architecture Performance**:
   - **Overall Accuracy ($\le 1$ px)** increased from **32.50% (ZNCC)** to **49.17%** (a **+16.67% absolute improvement**).
   - **Median Error** dropped from **40.42 px (ZNCC)** to **13.89 px** (a **65.6% reduction in median error**).
   - Significant accuracy gains across all difficulty tiers:
     - **Easy**: $70.0\% \rightarrow \mathbf{73.3\%}$
     - **Medium**: $40.0\% \rightarrow \mathbf{50.0\%}$
     - **Hard**: $13.3\% \rightarrow \mathbf{43.3\%}$ (+30.0% gain)
     - **Adversarial**: $20.0\% \rightarrow \mathbf{30.0\%}$ (+10.0% gain)
   - Zero deep-learning weights or GPU requirements needed, providing high explainability with subpixel precision.

---

## 2. Diagnostics: Clean Oracle & Top-K Spatial Recall

### 2.1 Level 1: Clean-Case Oracle Validation
Synthesized 20 noise-free, distortion-free, shear-free image pairs with $\text{scale} = 1.0$ and $\text{rotation} = 0^\circ$:
- **Pass Rate**: $75.0\%$ exact subpixel matches ($\le 0.3$ px error).
- **Error Modes in Remaining 25%**: In cases where the template was cropped from an un-patterned, perfectly uniform repeating periodic array, candidate peaks at adjacent fin/wordline pitches ($dx = \pm 48.4$ px, $dy = \pm 65.5$ px) produced mathematically identical correlation scores ($\Delta S < 0.005$).

### 2.2 Level 2: Top-K Candidate Spatial Recall (120 Benchmark Cases)

| Retrieval Method | Top-1 Recall (%) | Top-3 Recall (%) | Top-5 Recall (%) | Top-10 Recall (%) | Retrieval vs. Ranking Gap |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Intensity FFT-NCC** | 35.83% | 48.33% | 55.83% | 60.83% | **+25.00%** |
| **Gradient FFT-NCC** | 31.67% | 40.00% | 43.33% | 48.33% | **+16.67%** |
| **MultiScale Dual-Channel (Proposed)** | **38.33%** | **50.00%** | **54.17%** | **65.00%** | **+26.67%** |

#### Top-10 Candidate Recall by Difficulty Level:
- **Easy**: **80.0%**
- **Medium**: **76.7%**
- **Hard**: **60.0%**
- **Adversarial**: **43.3%**

---

## 3. Comprehensive Benchmark Comparison (120 Samples)

The complete benchmark evaluation was executed across the reproducible 120-sample dataset comprising 30 Easy, 30 Medium, 30 Hard, and 30 Adversarial image pairs spanning DRAM (`dram_1x`, `dram_dense`) and FinFET (`finfet_10nm`, `finfet_7nm`) architectures.

### Overall Performance Table

| Method / Variant | Acc ($\le 1$ px) % | Acc ($\le 3$ px) % | Acc ($\le 5$ px) % | Mean Err (px) | Median Err (px) | P95 Err (px) | Mean Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V0: Baseline ZNCC** | 32.50% | 35.83% | 35.83% | 143.18 | 40.42 | 635.21 | 193.68 ms |
| **V1: Baseline Plain FFT-NCC** | 31.67% | 35.00% | 35.83% | 172.13 | 43.33 | 844.89 | **56.37 ms** |
| **V2: Gradient FFT-NCC** | 26.67% | 31.67% | 31.67% | 235.92 | 104.70 | 781.54 | 67.01 ms |
| **V3: Multi-Scale Dual Channel** | 30.00% | 38.33% | 38.33% | 168.90 | 43.51 | 713.72 | 418.38 ms |
| **V4: Proposed Structural Registration** | **49.17%** | **49.17%** | **49.17%** | **121.89** | **13.89** | **628.42** | 1981.36 ms |

---

### Accuracy ($\le 5$ px) Breakdown by Difficulty

| Variant | Easy (%) | Medium (%) | Hard (%) | Adversarial (%) |
| :--- | :---: | :---: | :---: | :---: |
| **V0: Baseline ZNCC** | 70.0% | 40.0% | 13.3% | 20.0% |
| **V1: Baseline FFT-NCC** | 70.0% | 40.0% | 16.7% | 16.7% |
| **V2: Gradient FFT-NCC** | 70.0% | 46.7% | 10.0% | 0.0% |
| **V3: Multi-Scale Dual** | 70.0% | 33.3% | 26.7% | 23.3% |
| **V4: Proposed Structural Registration** | **73.3%** | **50.0%** | **43.3%** | **30.0%** |

---

### Accuracy ($\le 5$ px) Breakdown by Architecture

| Variant | DRAM 1x (%) | DRAM Dense (%) | FinFET 10nm (%) | FinFET 7nm (%) |
| :--- | :---: | :---: | :---: | :---: |
| **V0: Baseline ZNCC** | 34.4% | 34.4% | 32.1% | 42.9% |
| **V1: Baseline FFT-NCC** | 40.6% | 37.5% | 28.6% | 35.7% |
| **V2: Gradient FFT-NCC** | 31.2% | 34.4% | 32.1% | 28.6% |
| **V3: Multi-Scale Dual** | 31.2% | 37.5% | 35.7% | 50.0% |
| **V4: Proposed Structural Registration** | **46.9%** | **53.1%** | **39.3%** | **57.1%** |

---

## 4. Architecture Pipeline Description

The standalone inference pipeline (`inference.py`) implements the 7 stages:
1. **Noise-Adaptive Normalization**: Percentile-based robust contrast normalization and gentle Gaussian filtering to suppress SEM high-frequency shot noise without blurring fin/wordline edges.
2. **Multi-Scale & Micro-Rotation Search Bank**: Scale bank ($s \in [0.95, 0.98, 1.00, 1.02, 1.05]$) combined with bounded micro-rotations ($r \in [-3.0^\circ, -1.5^\circ, 0.0^\circ, 1.5^\circ, 3.0^\circ]$).
3. **Dual Structural Correlation**: Multi-channel response maps fusing Normalized Intensity ($S_I$) and Scharr Gradient Magnitude ($S_G$):
   $$S_{\text{combo}} = 0.55 \cdot S_I + 0.45 \cdot S_G$$
4. **Spatial Non-Maximum Suppression (NMS)**: Eliminates adjacent redundant pixels, extracting the top 15 distinct spatial peak candidates across the entire search field of view.
5. **Local Multi-Feature Candidate Verification**: For each candidate patch:
   - Evaluates patch structural alignment and gradient coherence ($S_{\text{Grad}}$).
   - Runs local 2D phase correlation with Hanning windowing to determine local translational shift and phase peak sharpness ($S_{\text{Phase}}$).
   - Re-ranks candidates via composite score:
     $$S_{\text{final}} = 0.55 \cdot S_{\text{combo}} + 0.30 \cdot S_{\text{Grad}} + 0.15 \cdot S_{\text{Phase}}$$
6. **2D Paraboloid Subpixel Refinement**: Fits a 2D quadratic paraboloid surface on the $5 \times 5$ local correlation neighborhood around the winning peak:
   $$z(x, y) = a x^2 + b y^2 + c x y + d x + e y + f$$
   Analytically resolving the true subpixel centroid:
   $$\delta x = \frac{c e - 2 b d}{4 a b - c^2}, \quad \delta y = \frac{c d - 2 a e}{4 a b - c^2}$$
7. **Problem-Statement Compliant CLI**: Accepts `--reference <path>` and `--search <path>`, outputting standard `(x, y)` format directly to stdout.
