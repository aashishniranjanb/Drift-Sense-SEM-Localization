# RETRIEVAL-V2 DETAILED MILESTONE REPORT

## Executive Summary
- **Baseline Recall (V25 Top-200):** 105 / 140 (75.0%)
- **Direct Candidate Recall (Top-800):** 119 / 140 (85.0%)  
- **Effective Local-Refinement Opportunity:** 131 / 140 (93.6%)  
- **Expanded 5th-Order Lattice (k=1..5) Coverage:** 133 / 140 (95.0% within 10px)  
- **Newly Retrieved Direct GT Candidates:** +14  
- **V54 Anchor Preservation:** 140 / 140 (100.0%)

---

## 1. Top-K Direct Candidate Recall Curve

| Top-K Candidate Limit | Present Hits (out of 140) | Recall (%) | Notes |
|---|---|---|---|
| Top 1 | 76 | 54.3% | V54 Baseline Top-1 Successes |
| Top 5 | 87 | 62.1% | |
| Top 10 | 93 | 66.4% | |
| Top 20 | 98 | 70.0% | |
| Top 50 | 104 | 74.3% | |
| Top 100 | 108 | 77.1% | |
| **Top 200** | **110** | **78.6%** | Baseline V25 Anchor Pool |
| Top 300 | 114 | 81.4% | |
| Top 500 | 117 | 83.6% | |
| **Top 800** | **119** | **85.0%** | **RETRIEVAL-V2 DIRECT RECALL** |

---

## 2. Source Attribution Breakdown (+14 New Candidates)

| Discovery Source | Pair IDs | Count |
|---|---|---|
| **Local Lattice Subpixel Probes** | `pair_014`, `pair_050`, `pair_076`, `pair_083`, `pair_119` | 5 |
| **Phase Correlation** | `pair_031`, `pair_075` | 2 |
| **Multi-Scale Context** | `pair_096` | 1 |
| **Combined Multi-Source** | `pair_000`, `pair_018`, `pair_034`, `pair_040`, `pair_042`, `pair_060` | 6 |

---

## 3. Effective 131 Subpixel Refinement Classification

| Classification Category | Count / 140 | Description |
|---|---|---|
| **DIRECT** | 113 | Candidate physically within $\le 5.0\text{px}$ error |
| **NEAR** | 8 | Candidate within $5.0\text{px} - 10.0\text{px}$ error |
| **REFINABLE ($\le 1\text{px}$)** | 84 | Parabolic fit brings coordinate within $\le 1.0\text{px}$ |
| **REFINABLE ($\le 5\text{px}$)** | 29 | Parabolic fit brings coordinate within $\le 5.0\text{px}$ |
| **NOT REFINABLE** | 27 | Residual error $> 5.0\text{px}$ after parabolic fit |
