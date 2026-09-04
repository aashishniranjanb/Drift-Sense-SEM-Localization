# DRIFT-SENSE SEM LOCALIZATION CHAMPIONSHIP ARCHIVE

**Archive Version:** 1.0.0
**Date:** 2026-09-04
**Authoritative Baseline:** V54 Scale Refined Golden Anchor @ **91.040 / 100**

---

## 1. Directory Structure Overview

```
DRIFT_SENSE_CHAMPIONSHIP_ARCHIVE/
│
├── 01_GOLDEN_BASELINE/
│   └── V54_SCALE_REFINED_91.040_VERIFIED/
│       ├── FINAL_SUBMISSION/            # Immutable production pipeline codebase
│       ├── BASELINE_MANIFEST.txt        # Full SHA256 file manifest
│       ├── SCORE_REPORT.json            # Exact 180-pair benchmark score breakdown
│       └── SHA256SUMS.txt               # Security verification checksums
│
├── 02_96PLUS_BASELINE/
│   └── TARGET_96PLUS/                   # Candidate folder for 96+ grayscale target
│       ├── SCORE_REPORT.md
│       ├── SCORE_BREAKDOWN.md
│       ├── VERIFICATION/
│       └── STATUS.md
│
├── 03_RGB_102PLUS/
│   └── TARGET_102PLUS/                  # Candidate folder for RGB bonus track
│       ├── RGB_VERIFICATION/
│       ├── SCORE_REPORT.md
│       ├── ELIGIBILITY_STATUS.md
│       └── SHA256SUMS.txt
│
├── 04_RESEARCH_REJECTED/                # Controlled research & audit logs
│   ├── RERANK_V2/                       # ML pairwise ranker (Rejected: 89.979)
│   ├── RERANK_V3/                       # Non-ML structural selector (Rejected: 91.040)
│   └── LATTICE_RESCUE_V1/               # Local lattice rescue (Rejected: +0.057, 3 broken)
│
└── ARCHIVE_MANIFEST.md                  # Master archive specification (This file)
```

---

## 2. Official Score Ladder & Evidence Labels

| Level | Name / Folder | Target / Score | Status | Key Condition / Evidence |
|---|---|---|---|---|
| **01** | `V54_SCALE_REFINED_91.040_VERIFIED` | **91.040 / 100** | **VERIFIED BASELINE** | 40/40 Loc, 19.74/20 Pose, 8.03/15 Rej, 8.27/10 Calib |
| **02** | `TARGET_96PLUS` | **96.000 / 100** | **OPTIMIZATION TARGET** | Requires resolving candidate ranking/retrieval failures |
| **03** | `TARGET_102PLUS` | **102.000 / 100** (Bonus-Adj) | **CONDITIONAL TARGET** | 96+ Grayscale Base + up to +6 RGB Bonus |

> [!IMPORTANT]
> Directories for unverified targets are strictly labeled `TARGET_96PLUS` and `TARGET_102PLUS`. They are renamed to `96PLUS_VERIFIED` or `102PLUS_VERIFIED` **only** after clean-room empirical scorer verification.

---

## 3. Immutable Golden Anchor Statement

The `V54_SCALE_REFINED_91.040_VERIFIED` codebase (`register.py`, `runtime/`, `models/`) is **immutable**. It serves as the authoritative production submission and rollback anchor. All experimental changes are developed in isolated branches or shadow scripts without mutating the golden anchor.

---

## 4. Research & Rejected Experiment Log

| Experiment | Target Hypothesis | Result | Key Forensic Finding | Verdict |
|---|---|---|---|---|
| **RERANK-V2** | ML pairwise tournament (LR/HGB) | 89.979 (-1.061) | Learned negative NCC weight; synthetic residual unrobust | **REJECTED** |
| **RERANK-V3** | Non-ML structural rule (K20-75) | 91.040 (+0.000) | Replicas lead in gradient NCC in 100% of ranking failures | **REJECTED** |
| **LATTICE_RESCUE_V1** | Local lattice vector rescue | 91.097 (+0.057) | Recovered pair_058 (101.72→0.26px), but broke 3 verified successes | **REJECTED** |

---

## 5. Optimization Tree & Roadmap

```
91.040 VERIFIED GOLDEN (V54 Anchor)
        │
        ├── Grayscale Retrieval / Re-ranking Optimization
        │        ↓
        │     96+ TARGET (TARGET_96PLUS)
        │
        └── RGB Extension Track
                 ↓
          +6 BONUS ELIGIBILITY (Set D)
                 ↓
             102+ TARGET (TARGET_102PLUS)
```
