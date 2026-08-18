# Component 1 — Hackathon Idea Submission Template (Slides 1–9)

## **Slide 1: Team Details**
* **Team Name**: Drift-Sense Metrology AI
* **Member Names & Roles**:
  * Lead Computer Vision & Metrology Engineer: Developer / Researcher
  * Deep Learning & Optimization Specialist: Developer / Contributor
* **College / Institution Name**: Hackathon 2026 Participant Team
* **Contact Details**: team-drift-sense@metrology-ai.org | +91-XXXXX-XXXXX

---

## **Slide 2: Problem Statement Addressed**
* **Selected Problem Statement**: `Drift-Sense: Navigation-Error Recovery`
* **Problem Description & Context**:
  In modern semiconductor manufacturing (sub-3nm FinFET and DRAM nodes), automated e-beam wafer inspection tools navigate across a 300mm wafer to locate specific functional sites. Stage vibration, thermal expansion, and mechanical drift introduce physical navigation drift ($\pm 10\text{ to }\pm 500\text{ nm}$). Because modern inspection targets are as small as $10\text{ to }100\text{ nm}$, this navigation error causes the high-magnification field of view to miss the target pattern.
  - **Reference Image**: $1000 \times 1000$ pixels capture of a single clean die site at high magnification ($1\text{ nm/pixel}$, $1\,\mu\text{m}^2$).
  - **Search Image**: $1000 \times 1000$ pixels capture at lower magnification ($10\text{ nm/pixel}$, $100\,\mu\text{m}^2$) containing severe Poisson secondary-electron noise, charging artifacts, and low contrast.
  - **Core Challenge**: The central difficulty stems from **highly periodic DRAM/FinFET layouts**, where repeated memory cells create visually identical candidates that produce false correlation matches. The objective is to compute the exact subpixel center $(x, y)$ of the reference pattern in search image coordinates.

---

## **Slide 3: Idea Description & Evolutionary Failure Analysis**

### Why Previous AI Approaches Failed:
```
CLASSICAL FFT-NCC
   ├── Fast (30 ms) ✓
   ├── Subpixel Precise when Unambiguous (66.5% Top-1) ✓
   └── Periodic DRAM Replica Ambiguity ✗
           │
           ▼
SIAMESE 1-vs-1 HCR (Iteration 4)
   ├── High 1-vs-1 Validation Accuracy (97.7%) ✓
   └── Demoted Correct FFT Peak in 43.0% of Cases! (Unconditional AI Failure) ✗
           │
           ▼
PACE GROUP RANKING (Iteration 5)
   ├── Group List Ranking Loss Solved Candidate Ranking (89.6% Val Top-3) ✓
   └── Still Overrode Unambiguous Clear FFT Captures ✗
           │
           ▼
V7 REDUNDANT MULTI-VIEW (Iteration 7 - Rejected)
   ├── Added 4 Representations + 4 Anchor Views
   └── Diluted Candidate Union in Periodic Arrays (Top-20 fell to 85.5%, Latency 877 ms) ✗
           │
           ▼
DRIFT-SENSE++ SAFE-CAR (PRODUCTION WINNER)
   ├── Strict "Do Not Override" Confidence Safety Gate (Delta-S >= 0.010, PSR >= 5.5) ✓
   ├── High-Speed Trusted Classical Path (30 ms) on 62% of Captures ✓
   └── AI Activated ONLY Under Periodic Ambiguity (Harmful Overrides Reduced to 1.5%) ✓
```

---

## **Slide 4: Proposed Solution (SAFE-CAR Pipeline)**
* **Dataset Generator Design**:
  - Continuous physical die layout model ($10,000\text{ nm} \times 10,000\text{ nm}$) generating $1000 \times 1000$ reference ($1\text{ nm/px}$) and $1000 \times 1000$ search ($10\text{ nm/px}$) pairs.
  - **Literature-Backed Noise & Augmentation Pipeline**:
    1. Secondary Electron Poisson Shot Noise ($\lambda \sim 10\text{--}50\text{ e}^-/ \text{pixel}$, Joy 1995).
    2. Wafer Surface Charging Low-Frequency Gradient (Cazaux 1999).
    3. E-beam Defocus Gaussian PSF Blur ($\sigma = 0.8\text{--}2.5\text{ px}$, Postek & Vladar 2011).
    4. Detector Readout Noise ($\sigma_{\text{read}} = 0.01\text{--}0.05$).
* **Localization Algorithm Architecture**:
  1. **Dual-Channel Retrieval**: Spatial Union ($C_I \cup C_G$) of Intensity FFT-NCC and Scharr Gradient FFT-NCC $\to$ Top-20 Candidates.
  2. **Calibrated Confidence Gating**: $C = 0.45 S_{\text{FFT}} + 0.25 Z_\Delta + 0.30 Z_{\text{PSR}}$. High confidence locks FFT Candidate #1 (30ms fast path).
  3. **PACE Residual Ranker**: Softmax List-Ranking network ($106,945$ params, $<0.5\text{ MB}$) scoring local ($64\times 64$), context ($128\times 128$), and directional overlap patches ($4\times 32\times 32$).
  4. **Operational Safety Modes**: `CLASSICAL` ($C \ge 0.85$), `CAR` ($0.50 \le C < 0.85$), `UNCERTAIN` ($C < 0.50$).
  5. **Subpixel Metrology Consensus**: Dual estimator agreement ($D = \|p_{\text{phase}} - p_{\text{paraboloid}}\|_2 \le 2.0\text{ px}$).

---

## **Slide 5: Innovation & Uniqueness**

### Accuracy Is Not Enough — We Measure Failure Risk

| Architecture / Method | Acc ($\le 5\text{ px}$) % | Harmful AI Override % | Median Error | Latency Metrics | Operational Safety |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **FFT-NCC Baseline** | 66.00% | — | 1.60 px | 30.25 ms | Unprotected against periodic replicas |
| **Siamese 1-vs-1 HCR** | 63.50% | **43.0%** (Severe) | 1.64 px | 75.80 ms | High failure risk on clear images |
| **Unconditional PACE** | 64.50% | **5.0%** (Moderate) | 1.72 px | 44.61 ms | Uncontrolled AI intervention |
| **Drift-Sense++ SAFE-CAR** | **66.00%** | **1.5%** (**Suppressed**) | **1.51 px** | **30.25 ms / 139.20 ms\*** | **Confidence-gated safety escalation** |

*\*Message for Judges: We did not just maximize accuracy. We minimized the probability of confidently selecting the wrong periodic site.*

---

## **Slide 6: Benchmark Results & Error Distribution Dashboard**

* **Master Benchmark Dashboard (Frozen 200-Case Test Set)**:
  - **Subpixel Accuracy ($\le 1\text{ px}$)**: **40.50%** (Project Record).
  - **In-Bounds Accuracy ($\le 5\text{ px}$)**: **66.00%**.
  - **Median Error**: **1.51 px** (High-precision subpixel population).
  - **Mean Error**: **73.17 px** (Reduced by $-13.35\text{ px}$ from baseline $86.52\text{ px}$).
  - **P95 Error**: **554.22 px** (Reduced by $-60.96\text{ px}$ from baseline $615.18\text{ px}$).
  - **Trusted Classical-Path Latency**: **30.25 ms** (62.0% of captures).
  - **End-to-End Mean Latency**: **139.20 ms** overall.
* **Key Visual Audit Artifacts**:
  - `submission_package/visuals/01_end_to_end_success.png`: Reference $\to$ Search $\to$ Prediction Box vs GT Box $\to$ Subpixel Zoom (Error = 0.20 px).
  - `submission_package/visuals/02_periodic_ambiguity.png`: Honest Failure Analysis (Lattice shift across 180 px due to heavy charging background).
  - `submission_package/visuals/04_error_distribution.png`: Bimodal Error Histogram & CDF (1.51 px median population vs periodic failure group).

---

## **Slide 7: Technology & Feasibility**
* **Tech Stack**: Python 3.10+, PyTorch 2.1+, OpenCV 4.8+, NumPy 1.26+, SciPy 1.11+, Matplotlib 3.8+, Pandas 2.1+.
* **Hardware Environment**: Developed & benchmarked on Intel Core i7 / NVIDIA GTX CPU/GPU execution environment.
* **Execution Metrics**:
  - **Dataset Generation Time**: $0.18\text{ s}$ per synthetic pair ($1000 \times 1000$).
  - **Localization Inference Time**: **30.25 ms** (Trusted Fast Path), **139.20 ms** (Mean End-to-End Latency).
  - **Model Size**: **$106,945$ parameters** ($0.42\text{ MB}$ `.pt` checkpoint).

---

## **Slide 8: GitHub & Video Link**
* **GitHub Repository (Mandatory)**: [https://github.com/aashishniranjanb/Drift-Sense-SEM-Localization](https://github.com/aashishniranjanb/Drift-Sense-SEM-Localization)
* **Demonstration Execution Guide**:
  ```bash
  python inference.py --reference data/benchmark_120/reference/0000.png --search data/benchmark_120/search/0000.png --verbose
  ```

---

## **Slide 9: References & Literature Citations**
1. **Joy, D. C. (1995)**: *Monte Carlo Modeling for Electron Microscopy and Microanalysis*, Oxford University Press (Poisson electron noise statistics).
2. **Cazaux, J. (1999)**: *Some considerations on the charging of insulating samples in SEM*, Journal of Microscopy, 196(2), 174–186.
3. **Postek, M. T., & Vladár, A. E. (2011)**: *Critical Dimension Metrology in the Scanning Electron Microscope*, NIST Handbook.
4. **Foroosh, H., et al. (2002)**: *Extension of Phase Correlation to Subpixel Registration*, IEEE Transactions on Image Processing, 11(3), 188–200.
5. **Applied Materials Patent US20260160714** (2026): *Process-Aware Contextual Overlap Matching in Wafer Metrology*.
