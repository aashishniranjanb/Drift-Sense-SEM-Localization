# Drift-Sense++ Architectural Decision Log

This log records the key architectural forks, empirical hypotheses, and formal decisions made across the V1 → V48 research trajectory.

---

## Log Entries

### [DEC-01] Coarse Search Decoupling (V10)
- **Problem:** Exhaustive 4-D parameter search over $(x, y, \theta, \text{scale})$ was computationally intractable (> 15 s/pair).
- **Options Considered:**
  1. Multi-resolution pyramid continuous optimizer.
  2. Coarse frequency-domain grid search + localized continuous subpixel refinement.
- **Decision:** **Adopt Option 2.** Log-polar coarse FFT search covers scale $[8\times, 12\times]$ and orientation $[\pm 5^\circ]$ in $< 0.05\text{ s}$, handing off to localized refinement.
- **Status:** **RETAINED (Production).**

---

### [DEC-02] 200-Candidate Pool vs. Single-Peak Greedy Search (V18)
- **Problem:** Single-peak greedy correlation trapped on adjacent cell replicas in 35% of degraded DRAM pairs.
- **Decision:** Extract a spatial pool of the top **200 local maxima** using spatial Non-Maximum Suppression (NMS) with physical lattice pitch radius.
- **Outcome:** Localization candidate recall jumped from 65% to **98.6%**.
- **Status:** **RETAINED (Production).**

---

### [DEC-03] Adaptive NMS Rejection (V26-A)
- **Hypothesis:** Dynamically adjusting NMS radius based on local peak density will improve candidate separation on irregular layouts.
- **Empirical Test:** Evaluated on all 180 development pairs.
- **Outcome:** Localization score dropped from 40.00 to 31.50. Dense clusters suppressed the true physical peak.
- **Decision:** **REJECTED.** Fixed physical lattice radius is strictly safer than adaptive density clustering.
- **Status:** **REJECTED (Dead End).**

---

### [DEC-04] Pairwise Tournament Ranking (V26-B)
- **Hypothesis:** Training a pairwise classifier ($C_i \text{ vs } C_j$) will resolve subtle differences between adjacent replicas.
- **Empirical Test:** Tested on 70 Set B degraded pairs.
- **Outcome:** Runtime exploded 8x; cyclic preferences ($A > B > C > A$) contaminated ranking on periodic structures.
- **Decision:** **REJECTED.** Pointwise multi-evidence scoring is globally consistent and deterministic.
- **Status:** **REJECTED (Dead End).**

---

### [DEC-05] Deep Neural Feature Embeddings (V40)
- **Hypothesis:** Fine-tuning MobileNet/ResNet embeddings will learn noise-invariant representations for SEM images.
- **Empirical Test:** Evaluated feature map alignment on 10 nm FinFET patterns.
- **Outcome:** Convolutional striding and pooling destroyed subpixel edge phase; localization error degraded to $\ge 2.4\text{ px}$.
- **Decision:** **REJECTED.** Frequency-domain FFT-NCC and steerable Sobel gradient filters preserve exact subpixel spatial phase.
- **Status:** **REJECTED (Dead End).**

---

### [DEC-06] Conservative Rejection vs. Aggressive Candidate Rescue (V44–V48)
- **Problem:** 64 present pairs were classified as "false negatives" due to strict rejection thresholds.
- **Experiments (V44–V46):** Lowered presence thresholds to aggressively rescue candidates. While 11 present pairs were rescued, **24 absent pairs were falsely accepted** as present on Set C.
- **Analysis:** In the competition scoring rubric:
  - Rescuing 11 present pairs adds $\approx +2.8\text{ points}$.
  - Adding 24 absent false positives deducts $-7.6\text{ points}$ from Rejection ($15\text{ pts}$ category).
  - Net score delta was negative.
- **Decision:** **RETAINED CONSERVATIVE GATE.** Strict rejection of ambiguity mathematically dominates risky retrieval. Only 1 pair (`pair_045`) was safely rescued without risking false accepts.
- **Status:** **RETAINED (Production Baseline Frozen at 90.50).**
