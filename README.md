# Drift-Sense++ SAFE-CAR: Adaptive Structural SEM Wafer Localization Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![OpenCV 4.8+](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Drift-Sense++ SAFE-CAR (Structural Ambiguity-aware Failure-aware Escalation)** is an ultra-fast, subpixel-accurate SEM wafer localization framework engineered for Applied Materials' *Drift-Sense: Navigation-Error Recovery* challenge.

---

## Quickstart - Applied Materials Judge Execution

### 1. Installation
```bash
git clone https://github.com/aashishniranjanb/Drift-Sense-SEM-Localization.git
cd Drift-Sense-SEM-Localization
pip install -r requirements.txt
```

### 2. Standalone Inference CLI Test
```bash
python inference.py --reference demo/reference.png --search demo/search.png --verbose
```

#### Expected CLI Output:
```json
{
  "x": 305.09,
  "y": 620.88,
  "confidence_score": 0.7654,
  "mode": "CLASSICAL",
  "decision": "LOCALIZED",
  "uncertainty": "LOW",
  "status": "OK",
  "path": "FAST_TRUSTED_FFT",
  "latency_ms": 34.88
}
(305.09, 620.88)
```

### 3. Standalone Synthetic Dataset Generator
```bash
python dataset_generator.py --architecture DRAM --num-pairs 20 --output-dir demo_data/
```

---

## Performance Benchmark Summary (Frozen 200-Case Held-Out Test Set)

| Metric | Metric Definition & Benchmark Value |
| :--- | :--- |
| **Subpixel Precision (<= 1 px)** | **40.5%** of the 200 held-out cases were localized within <= 1 px |
| **In-Bounds Accuracy (<= 5 px)** | **66.0%** of the 200 held-out cases were localized within <= 5 px |
| **Median Error** | **1.51 px** median localization error |
| **P95 Error** | **554.22 px** P95 error *(Bimodal failure distribution)* |
| **Trusted Fast-Path Latency** | **30.25 ms** *(Executed on 62.0% of captures)* |
| **End-to-End Mean Latency** | **139.20 ms** overall end-to-end latency |
| **Harmful AI Overrides** | **1.5%** *(Suppressed from 43.0% in binary models)* |

---

## Architectural Evolution & Scientific Research Story

```
V1 (ZNCC / FFT-NCC) -> Fast (30 ms), but vulnerable to periodic DRAM cell ambiguity
       |
V2 (Multi-Scale Dual) -> Robust, but unacceptable compute latency (1.98 s)
       |
V3 (Adaptive Gated) -> Routing alone insufficient (hard path accuracy 30.8%)
       |
V4 (Siamese 1-vs-1 HCR) -> 97.7% binary val acc, BUT demoted GT in 43.0% of cases
       |
V5 (Unconditional PACE) -> Group List Ranking solved ranking, BUT overrode clear FFT
       |
V6 (SAFE-CAR Winner) -> Confidence Gate (Delta-S >= 0.010, PSR >= 5.5) suppressed AI overrides to 1.5%
       |
V7 (Multi-View Retrieval) -> Added 4 features + 4 anchors, BUT diluted candidate union (Top-20 fell to 85.5%)
       |
       v
DRIFT-SENSE++ SAFE-CAR: Frozen Production Winner & Empirical Pareto Frontier
```

*Scientific Conclusion: V7 was rejected by a pre-defined acceptance gate, establishing SAFE-CAR as the true empirical Pareto frontier.*

---

## High-Impact Visual Artifacts Map

1. **End-to-End Success Visualization**: `submission_package/visuals/01_end_to_end_success.png`
2. **Periodic Ambiguity Shift Analysis**: `submission_package/visuals/02_periodic_ambiguity.png`
3. **Confidence Safety Gate Diagram**: `submission_package/visuals/03_confidence_gate.png`
4. **Bimodal Error Histogram & CDF**: `submission_package/visuals/04_error_distribution.png`
5. **V1-V7 Scientific Progression**: `submission_package/visuals/05_ablation_comparison.png`
6. **End-to-End System Overview**: `submission_package/visuals/06_system_overview.png`
7. **Perturbation Stress Matrix Heatmap**: `submission_package/visuals/stress_matrix_heatmap.png`

---

## License & Citation
- **License**: MIT License
- **Literature References**: Joy (1995), Cazaux (1999), Postek & Vladar (2011), Foroosh (2002), Applied Materials Patent US20260160714 (2026). See `submission_package/REFERENCES_CITATIONS.md`.
