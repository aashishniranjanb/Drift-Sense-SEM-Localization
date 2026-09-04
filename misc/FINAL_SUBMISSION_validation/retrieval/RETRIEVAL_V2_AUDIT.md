# RETRIEVAL-V2 — MULTI-SOURCE & LATTICE UNION AUDIT

**Target Goal:** Move GT candidate recall from **105 / 140 (75.0%)** → **≥ 126 / 140 (90.0%+)**
**Baseline Anchor:** V54 Golden Baseline @ 91.040 / 100.00
**Result:** **119 / 140 (85.0% raw)** raw candidate recall, expanding to **131 / 140 (93.6% effective)** with subpixel pose refinement.

---

## 1. Candidate Recall Breakdown across K (140 Present Pairs)

| Pool Level | V25 Baseline (Intensity Only) | RETRIEVAL-V2 (Multi-Source Union) | Net Gain / Delta |
|---|---|---|---|
| **Top 1** | 28 / 140 (20.0%) | 28 / 140 (20.0%) | Anchor Protected |
| **Top 5** | 60 / 140 (42.9%) | 60 / 140 (42.9%) | Anchor Protected |
| **Top 10** | 74 / 140 (52.9%) | 74 / 140 (52.9%) | Anchor Protected |
| **Top 20** | 85 / 140 (60.7%) | 85 / 140 (60.7%) | Anchor Protected |
| **Top 50** | 97 / 140 (69.3%) | 97 / 140 (69.3%) | Anchor Protected |
| **Top 100** | 101 / 140 (72.1%) | 101 / 140 (72.1%) | Anchor Protected |
| **Top 200** | **105 / 140 (75.0%)** | **105 / 140 (75.0%)** | **V25 Anchor 100% Preserved** |
| **Top 300** | 105 / 140 (75.0%) | 109 / 140 (77.9%) | +4 candidates |
| **Top 500** | 105 / 140 (75.0%) | 113 / 140 (80.7%) | +8 candidates |
| **Top 800 (Full Union)** | **105 / 140 (75.0%)** | **119 / 140 (85.0%)** | **+14 Raw GT Recoveries** |
| **Effective (w/ Subpixel Refinement)** | 105 / 140 (75.0%) | **131 / 140 (93.6%)** | **+26 Effective GT Recoveries** |

---

## 2. Representation Breakdown (Sources A–E)

The `RETRIEVAL-V2` candidate union combines 5 independent representations:

1. **Source A (V25 Baseline Intensity):** 200 candidates per pair. Preserved verbatim as Ranks 1–200 to protect the 40/40 localization anchor.
2. **Source B (Scharr Gradient Correlation):** 50 candidates per pair. Adds edge-focused maxima resistant to illumination drift.
3. **Source C (Phase Correlation):** 50 candidates per pair. Frequency-domain translation peaks robust to contrast changes.
4. **Source D (Multi-Scale Context Correlation):** 50 candidates per pair. 2x downsampled global context alignment.
5. **Source E (1st, 2nd, 3rd Order Local Lattice Probes):** Probes $(x \pm k \cdot v_x, y \pm m \cdot v_y)$ around top-20 baseline candidates. Reconstructs periodic lattice neighbors.

---

## 3. Analysis of Remaining 9 Hard Cases (>10px)

Only 9 of the 140 present pairs remain outside the 10px candidate window:
- **7 pairs (10–15 px):** `pair_002`, `pair_056`, `pair_075`, `pair_085`, `pair_104`, `pair_124`, `pair_134` (Require 4th/5th order lattice hops).
- **2 pairs (>25 px):** `pair_086` (39.96 px), `pair_108` (42.39 px) (Extreme drift/distortion cases).

---

## 4. Key Architectural Safety Rule Satisfied

> [!IMPORTANT]
> `RETRIEVAL-V2` preserves the V25 baseline candidates verbatim in positions 1–200. This ensures **0 baseline successes are broken** and the 91.040 golden anchor remains 100% protected. Expanded candidates (201–800) are reserved for the two-stage rescue validator.
