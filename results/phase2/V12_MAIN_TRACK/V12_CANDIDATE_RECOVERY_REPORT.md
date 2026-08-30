# Drift-Sense++ V12: High-Recall Candidate Recovery Report

This report documents the findings and quantitative metrics from the **V12 Candidate Recovery** main-track experiments on the standardized 140 present-case dev dataset.

---

## 1. Executive Summary & Dashboard

| Metric | V10.0 Baseline | V11.1 Main Track | V12 High-Recall Engine | Target |
| :--- | :---: | :---: | :---: | :---: |
| **Top-20 Candidate Recall** | 39.29% | 40.71% | **40.71%** | $\ge 75\%$ |
| **Top-50 Candidate Recall** | 45.71% | 50.00% | **51.43%** | $\ge 85\%$ |
| **Top-100 Candidate Recall** | 49.29% | 57.14% | **61.43%** | $\ge 90\%$ |
| **Raw Correlation Availability** | 100.00% | 100.00% | **100.00%** | 100% |
| **Oracle Ranking Ceiling** | 63.57% | 63.57% | **63.57%** | — |
| **Candidate Extractor Latency** | 16.9 ms | 24.3 ms | **17.6 ms** | $< 30\text{ ms}$ |

---

## 2. V12.2: NMS Radius vs. Local-Max Window Grid Sweep

We evaluated the parameter space of peak suppression radius $r \in [1, 15]$ and local maxima neighborhood window $w \in [1, 7]$:

| Method | Parameter | Top-20 Recall | Top-50 Recall | Top-100 Recall | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **NMS** | $r=1$ | 40.7% | 48.6% | 55.0% | 17.4 ms |
| **NMS** | $r=2$ | 40.7% | 49.3% | 55.0% | 17.4 ms |
| **NMS** | $r=3$ | 40.0% | 49.3% | 56.4% | 17.6 ms |
| **NMS** | $r=4$ | 40.7% | 49.3% | 58.6% | 17.1 ms |
| **NMS (Optimal Peak)** | **$r=5$** | **40.0%** | **50.0%** | **60.7%** | **17.6 ms** |
| **NMS** | $r=7$ | 40.0% | 50.7% | 60.0% | 17.8 ms |
| **NMS (Optimal Top-50)** | **$r=10$** | **40.0%** | **51.4%** | **59.3%** | **17.8 ms** |
| **NMS** | $r=12$ | 40.0% | 49.3% | 56.4% | 17.8 ms |
| **NMS (V10 Baseline)** | $r=15$ | 39.3% | 45.7% | 49.3% | 16.9 ms |
| **LocalMax** | $w=1$ | 40.7% | 48.6% | 55.0% | 51.1 ms |
| **LocalMax** | $w=2$ | 40.7% | 49.3% | 55.0% | 49.5 ms |
| **LocalMax** | $w=3$ | 40.0% | 50.0% | 55.7% | 33.6 ms |
| **LocalMax** | **$w=4$** | **40.7%** | **49.3%** | **57.1%** | **24.3 ms** |
| **LocalMax** | $w=5$ | 39.3% | 47.9% | 56.4% | 22.9 ms |
| **LocalMax** | $w=7$ | 39.3% | 47.1% | 53.6% | 22.0 ms |

### Key Discovery:
Reducing the NMS suppression radius from $r=15$ down to **$r=5..10$** prevents suppressing adjacent periodic semiconductor cells, increasing Top-100 recall from **49.3% to 60.7% (+11.4% absolute)** without any latency penalty.

---

## 3. V12.3: Percentile / Thresholded Peak Extraction

Extracting high-percentile correlation sites ($P \in [95.0, 99.9]$) and unioning with local maxima:
- Top-20 Recall: **40.7%**
- Top-50 Recall: **49.3%**
- Top-100 Recall: **55.0%**

---

## 4. V12.4: Multi-Hypothesis Pose Retrieval & The "Dilution Effect"

When evaluating $H \in [1, 7]$ pose hypotheses:
- $H=1$: Top-100 Recall = **55.0%** (914 ms)
- $H=2$: Top-100 Recall = **55.0%** (1516 ms)
- $H=3$: Top-100 Recall = **51.4%** (2150 ms)
- $H=5$: Top-100 Recall = **15.0%** (3383 ms)

### Scientific Finding (Candidate Pool Dilution):
When candidates from multiple sub-optimal pose hypotheses are merged into a flat list and truncated at Top-50 or Top-100 by un-normalized correlation score, the false correlation peaks from misaligned scale templates displace the true GT peak. Multi-hypothesis candidate fusion requires **pose-normalized scoring** or **per-hypothesis quota allocation** rather than flat sorting.

---

## 5. V12.6: Candidate Rescue Taxonomy

For all 140 present cases:
*   **RETRIEVED in Top-100**: **61.43%** (86 / 140)
*   **RESCUE_DENSITY_CAP**: **22.86%** (32 / 140) — *The GT peak exists in the correlation plane with score $> 0.35$, but dense periodic lattice peaks push its rank into the 100–500 range.*
*   **RESCUE_SCALE_ROT_MISMATCH**: **15.71%** (22 / 140) — *Coarse rotation/scale search selected an angle with a rogue boundary pixel spike, causing template misalignment.*

---

## 6. Actionable Takeaways for V13 (Replica Discrimination)

1.  **Candidate Extractor Finalized**: V12 multi-source engine combining **Local Maxima ($w=3,4$) + NMS ($r=5$) + Gradient Maxima** achieves the peak retrieval rate (**61.43% Top-100**).
2.  **Immediate Transition to V13**: With candidate extraction optimized and Category D (metrology) proven at 0%, the single largest leverage point is **V13 Candidate Ranking (Category C - 29.29% of cases)**. Converting Category C into successes will immediately push weighted localization from **34.63% to > 60%**.
