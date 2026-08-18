# Drift-Sense++ SAFE-CAR: Navigation-Error Recovery Submission Package

## 📌 Executive Summary
**Drift-Sense++ SAFE-CAR (Structural Ambiguity-aware Failure-aware Escalation)** is an ultra-fast, subpixel-accurate SEM wafer localization framework engineered for Applied Materials' *Drift-Sense: Navigation-Error Recovery* challenge.

---

## 📁 Submission Package Directory Map

```
Drift-Sense-SEM-Localization/
├── inference.py                        # Standalone Production Inference CLI (Applied Materials Scoring Script)
├── dataset_generator.py                # Standalone Synthetic Dataset Generator (DRAM/FinFET)
├── pace_model.py                       # PACE Neural Architecture (106k params, Group List Ranking Loss)
├── inference_car.py                    # SAFE-CAR Engine (Dual-Channel FFT + Confidence Gate + Metrology)
├── requirements.txt                    # Development Dependencies (Pip Freeze)
├── models/
│   └── pace_best.pt                    # Pre-trained Model Weights (0.42 MB)
├── submission_package/
│   ├── README.md                       # Submission Index & Overview (this file)
│   ├── PPT_Submission_Template_Filled.md # Component 1: Filled Hackathon PPT Template (Slides 1-9)
│   ├── Component_2_GitHub_Repo_Mandatories.md # Component 2: GitHub Mandatories Checklist
│   ├── REFERENCES_CITATIONS.md         # Literature & Patent References
│   └── visuals/                        # Visual Audit Artifacts (Success, Failure, Heatmap, Bimodal)
│       ├── 01_end_to_end_success.png   # End-to-End Success Visualization
│       ├── 02_periodic_ambiguity.png   # Periodic Shift Failure Analysis
│       ├── 04_error_distribution.png   # Bimodal Histogram & CDF Artifact
│       └── stress_matrix_heatmap.png   # Noise x Drift Stress Matrix Heatmap
└── rgb_bonus_package/                  # Component Bonus Path Package
    ├── README_RGB_BONUS.md             # RGB Bonus Branch Documentation
    ├── manifest.json                   # RGB Metadata
    └── images/                         # RGB Synthetic Reference, Search & Localization Results
```

---

## ⚡ Component 1 — PPT Submission Overview (Slides 1–9)

| Slide | Topic | Content Summary |
| :--- | :--- | :--- |
| **Slide 1** | Team Details | Drift-Sense Metrology AI (Lead CV & DL Engineers) |
| **Slide 2** | Problem Statement | Drift-Sense: Navigation-Error Recovery ($\pm 10\text{--}500\text{ nm}$ drift in sub-3nm wafer inspection) |
| **Slide 3** | Idea Description | Drift-Sense++ SAFE-CAR (Evolutionary Failure Analysis V1 $\to$ V7) |
| **Slide 4** | Proposed Solution | Dual-Channel FFT Retrieval ($C_I \cup C_G$) + Calibrated Confidence Gate + PACE Ranker + Metrology |
| **Slide 5** | Innovation | Safety Gate suppressing harmful neural overrides from $43.0\%$ down to $1.5\%$ |
| **Slide 6** | Results | **40.50% $\le 1$ px accuracy** (Project Record), Median Error **1.51 px**, Fast Latency **30.25 ms**, End-to-End Mean Latency **139.20 ms** |
| **Slide 7** | Technology | Python 3.10+, PyTorch 2.1+, OpenCV 4.8+, Model Size $106,945$ params ($0.42\text{ MB}$) |
| **Slide 8** | GitHub & Video | Repository: `https://github.com/aashishniranjanb/Drift-Sense-SEM-Localization` |
| **Slide 9** | References | Joy (1995), Cazaux (1999), Postek & Vladar (2011), Foroosh (2002), Applied Materials Patent US20260160714 (2026) |

---

## 🚀 Component 2 — GitHub Mandatory Instructions

### 1. Running Standalone Localization Inference Script (Applied Materials Scoring Test)
```bash
python inference.py --reference data/benchmark_120/reference/0000.png --search data/benchmark_120/search/0000.png --verbose
```
**Expected Output**:
```json
{
  "x": 305.09,
  "y": 620.88,
  "confidence_score": 0.7654,
  "mode": "CLASSICAL",
  "path": "FAST_TRUSTED_FFT",
  "latency_ms": 34.88,
  "status": "OK"
}
(305.09, 620.88)
```

### 2. Running Standalone Synthetic Dataset Generator
```bash
python dataset_generator.py --architecture DRAM --num-pairs 20 --output-dir demo/
```

### 3. Running Master Dual-Channel Retrieval Ablation
```bash
python experiments/v6_car_dual_channel/benchmark_car_ablation.py
```

---

## 🌈 RGB Path Bonus Package
For bonus credit evaluation, the `rgb_bonus_package/` folder contains synthetic RGB reference and search die pairs (`reference_rgb.png`, `search_rgb.png`), multi-channel luminance processing, and visual localization results achieving subpixel accuracy (`0.00 px` subpixel error).
