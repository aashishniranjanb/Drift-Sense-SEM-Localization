# Drift-Sense++ V8: 9-Phase Research & Benchmark Roadmap

This document serves as the authoritative research log and architectural specification for **Drift-Sense++ V8 (Phase 2)**. 

---

## Overview of Strategy

Phase 2 rewards a multi-objective evaluation:
1. **Localization (40 pts)**: Subpixel coordinate estimation $(x, y)$. Evaluated on $0.45 \times \text{Set A} + 0.55 \times \text{Set B}$.
2. **Pose Recovery (20 pts)**: Scale factor $s \in [8, 12]$ and Rotation angle $\theta \in [-5^\circ, +5^\circ]$.
3. **Rejection / Presence Detection (15 pts)**: Identifying if the reference is present (`found = 1`) or absent (`found = 0`, Set C).
4. **Confidence Calibration (10 pts)**: Internal monotonicity of confidence scores against correctness.
5. **Efficiency (5 pts)**: Target median latency $< 5\text{ s}$ per pair (soft goal $< 1\text{ s}$).
6. **Generator & Failure Analysis (10 pts)**: Realistic synthetic dataset generator & failure taxonomy.

---

## Summary Table of the 9 Phases

| Phase | Version | Key Innovation | Target Failure Mode / Critical Issue | Primary Metrics Tracked |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **V8.0** | Preliminary 20-Case Baseline Archival | Establishing initial code baseline | $F1=0.973$, $\text{MAE}_s=0.0306$, $\text{MAE}_\theta=0.0629^\circ$, $\le 5\text{px}=33.3\%$ |
| **2** | **V8.1** | Standardized 180-Case Generator & Failure Taxonomy | Unrealistic cross-architecture Set C & unsegmented A/B testing | Set A $\le 5\text{px}$, Set B $\le 5\text{px}$, Set C $F1$, Failure Taxonomy CSV |
| **3** | **V8.2** | Multi-Scale Context Verification ($32^2, 64^2, 128^2$) | Periodic cell identity confusion (local ambiguity) | Candidate Context Similarity, Top-1 Replica Resolution |
| **4** | **V8.3** | Dual-Channel Candidate Consensus | Single-channel false correlation peaks | Channel Disagreement Penalty, Top-1 Accuracy |
| **5** | **V8.4** | Local Phase Correlation Consistency | Spatial alignment drift & candidate displacement | Phase-FFT Peak Displacement Error |
| **6** | **V8.5** | Periodic Lattice Replica Detector | Systematic DRAM/FinFET periodic replica grid matches | Periodicity Ambiguity Index, Spatial Grid Detection |
| **7** | **V8.6** | Conditional PACE Context Re-Ranking | High-ambiguity replica tie-breaking | Ambiguity-Gated Re-Ranking Precision |
| **8** | **V8.7** | Monotonic Isotonic Score Calibration | Uncalibrated raw confidence scores ($\rho=0.391$) | Spearman Rank Monotonicity ($\rho$), Brier Score |
| **9** | **V8.8** | Set D RGB Branch & Final Hardening | Bonus Set D compliance & submission package packaging | Set D Credit, Submission CLI Package |

---

## Detailed Specifications per Phase

### Phase 1: V8.0 — Preliminary Baseline Archival
- **What it does**: Establishes the initial Phase 2 baseline using the 20-case preliminary sweep.
- **Critical Issue Addressed**: Historical logging and validation of the initial scale search ($s \in [8, 12]$) and rotation search ($\theta \in [-5^\circ, +5^\circ]$).
- **Technical Design**: Coarse-to-fine FFT search for scale/rotation, basic rejection thresholds.
- **Artifact**: `results/phase2/V8_BASELINE_20CASE.md`.

---

### Phase 2: V8.1 — Standardized 180-Case Generator & Failure Taxonomy
- **What it does**: Builds a realistic, standardized 180-case dataset (70 Set A Nominal, 70 Set B Degraded, 40 Set C Same-Architecture Hard Negatives).
- **Critical Issue Addressed**: Fixes cross-architecture Set C cheat and decouples Set A vs Set B evaluation. Implements automated failure classification taxonomy (`ABSENCE_FALSE_POSITIVE`, `PERIODIC_REPLICA`, `SCALE_ERROR`, etc.).
- **Technical Design**: `phase2/generate_phase2_dataset.py` & `phase2/benchmark_phase2.py`.

---

### Phase 3: V8.2 — Multi-Scale Context Verification
- **What it does**: Extracts 3 concentric multi-scale context descriptors ($32 \times 32$ local cell, $64 \times 64$ structure, $128 \times 128$ neighborhood) around each candidate.
- **Critical Issue Addressed**: Solves periodic cell identity confusion by evaluating surrounding structural context rather than isolated periodic cell intensity.
- **Technical Design**: `phase2/context_matcher.py`.

---

### Phase 4: V8.3 — Dual-Channel Candidate Consensus
- **What it does**: Computes correlation planes independently on Intensity and Scharr Gradient channels.
- **Critical Issue Addressed**: Penalizes candidates where Intensity FFT and Gradient FFT disagree on spatial candidate location.
- **Technical Design**: `phase2/channel_consensus.py`.

---

### Phase 5: V8.4 — Local Phase Correlation Consistency
- **What it does**: Runs lightweight phase correlation on Top-K candidates prior to final selection.
- **Critical Issue Addressed**: Filters out false correlation peaks where phase correlation displacement conflicts with FFT-NCC peak location.
- **Technical Design**: `phase2/phase_verifier.py`.

---

### Phase 6: V8.5 — Periodic Lattice Replica Detector
- **What it does**: Analyzes candidate spatial distribution to detect if candidates form a periodic 2D lattice.
- **Critical Issue Addressed**: Computes a Periodicity Ambiguity Index to flag periodic cell families and prevent blind selection of peak #1.
- **Technical Design**: `phase2/periodicity_detector.py`.

---

### Phase 7: V8.6 — Conditional PACE Context Re-Ranking
- **What it does**: Invokes the pre-trained PACE neural residual ranker strictly when Periodicity Ambiguity Index $> \tau$.
- **Critical Issue Addressed**: Prevents neural network override on unambiguous images while applying deep context re-ranking to hard periodic arrays.
- **Technical Design**: `phase2/conditional_pace.py`.

---

### Phase 8: V8.7 — Monotonic Isotonic Score Calibration
- **What it does**: Trains an Isotonic Regression mapping multi-evidence features to a monotonic confidence score $C \in [0, 1]$.
- **Critical Issue Addressed**: Replaces manual weighted sums with calibrated non-parametric monotonic mapping to maximize Spearman rank correlation ($\rho > 0.80$).
- **Technical Design**: `phase2/calibration.py`.

---

### Phase 9: V8.8 — Set D RGB Branch & Final Hardening
- **What it does**: Adds an optional luminance/chrominance branch for Set D optical RGB images and hardens `register.py` for submission.
- **Critical Issue Addressed**: Unlocks the +10 Set D bonus points without destabilizing grayscale performance.
- **Technical Design**: `phase2/register.py` & `phase2/rgb_branch.py`.
