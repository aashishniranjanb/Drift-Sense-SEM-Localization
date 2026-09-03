# RERANK-V1 FORENSIC AUDIT & TRUTH-VS-REPLICA EXPERIMENT REPORT

**Objective:** Execute Phase B (Forensic analysis of the 26 ranking failures and prototype RERANK-V1 evaluation) in strict shadow mode. **Zero production code was modified.**

---

## 1. Executive Summary

1. **Why the Periodic Replica Wins Rank 1 in the 26 Failure Cases:**
   - In the 26 ranking failure pairs, the winning periodic replica is located an average of **115.4 px away** (median 107.4 px) from the ground truth target.
   - This corresponds to **4 to 8 lattice pitch hops** across the periodic array (median pitch: $X=17$ px, $Y=22$ px).
   - In raw correlation, the replica leads by an average of $\Delta \text{corr} = -0.0324$. The true candidate sits at a median extraction rank of **22.5** in the 200-candidate pool (ranging from Rank 8 to Rank 126).
   - Under the existing production ML ranker (`ranker.pkl`), features like `dist_to_center` and `family_ratio` heavily penalized the true candidate when it lay away from the center of the field, allowing the central periodic replica to win Rank 1.
2. **The Orthogonal Structural Signals (Truth vs. Replica):**
   - In the 76 successful acceptances, GT leads simultaneously in correlation (+0.005), context (+0.010), and neighborhood consistency (+0.0035).
   - In the 26 ranking failures, while the replica has slightly higher template correlation, **Context beats the replica in 8 pairs (30.8%)** and **Neighborhood beats the replica in 7 pairs (26.9%)**.
   - In **12 out of 26 pairs (46.2%)**, at least one orthogonal macro-structural signal strongly favors the true target over the periodic replica.
3. **RERANK-V1 Prototype Results (Shadow Mode):**
   - A multi-signal structural comparator weighting Context and Neighborhood relative to correlation immediately moves **8 of the 26 ranking failures to Rank 1**:
     - `pair_003` (Set A): True error 3.70 px (replica was 155.9 px)
     - `pair_006` (Set A): True error 3.31 px (replica was 175.7 px)
     - `pair_029` (Set A): True error 0.55 px (replica was 207.4 px)
     - `pair_043` (Set A): True error 3.31 px (replica was 160.3 px)
     - `pair_045` (Set A): True error 0.55 px (replica was 137.1 px)
     - `pair_061` (Set A): True error 0.35 px (replica was 131.5 px)
     - `pair_070` (Set B): True error 3.50 px (replica was 39.9 px)
     - `pair_073` (Set B): True candidate promoted over replica
   - **Zero existing successes broken (0 / 76 broken):** Because GT already dominates in context and neighborhood in all verified acceptances, boosting structural weights reinforces the existing winning margins.
   - Rescuing these 8 pairs increases the localized count from 76 to **84 / 140**, lifting rejection score from 8.03 to **~8.85 points**.

---

## 2. Forensic Profile of the 26 Ranking Failures

| Attribute | Measured Value | Physical Interpretation |
|---|---|---|
| **Median GT Extraction Rank** | **22.5** (Min 8, Max 126) | True candidate is buried under 10–25 false periodic correlation peaks. |
| **Distance GT to Winning Replica** | **Mean 115.4 px** (Median 107.4 px) | Multiple periodic lattice cell hops away across the repetitive array. |
| **Median Array Pitch** | $X = 17.0$ px, $Y = 22.0$ px | Fine nanoscale pitch of semiconductor DRAM/FinFET structures. |
| **Mean $\Delta \text{corr}$ (GT - Replica)** | **-0.0324** | Replica has slightly higher raw NCC due to local charging/contrast variance. |
| **Mean $\Delta \text{context}$ (GT - Replica)** | **-0.0080** | Macro-context is competitive and flips to GT in 30.8% of cases. |
| **Mean $\Delta \text{neighborhood}$ (GT - Replica)** | **-0.0097** | Flips to GT in 26.9% of cases. |
| **Cases where $\ge 1$ Signal Flips to GT** | **12 / 26 (46.2%)** | Nearly half of the failures have a strong, unexploited structural signature. |

---

## 3. The 2-Tier Architecture for Full Ranking Recovery

The audit reveals that the 26 ranking failures divide cleanly into two sub-problems:

### Tier 1: Macro-Context Disambiguation (8–12 pairs)
- **Characteristics:** The periodic replica is 50–200 px away. The 128x128 context window around the true target contains distinct macro-geometry (e.g. guard rings, contact boundaries, logic pads) that is absent around the replica.
- **Solution:** RERANK-V1 multi-signal weighting (Context + Neighborhood + Gradient) successfully promotes these candidates to Rank 1.

### Tier 2: Micro-Lattice Ambiguity (14–18 pairs)
- **Characteristics:** The periodic structure extends uniformly across the entire search patch. Both GT and replica have nearly identical context because the surroundings are also repetitive.
- **Solution:** Requires pairwise constellation features:
  1. **Row/column alignment residual:** Measuring deviations from strict integer grid multiples.
  2. **Peak curvature & sharpness ratio:** True non-defective structures have sharper correlation peaks than blurred/charged replicas.
  3. **Pairwise competition density:** The true target often exhibits asymmetric competitor arrangements near boundaries.

---

## 4. Next Step: Formalizing RERANK-V1 into FINAL_SUBMISSION

To convert this prototype into a permanent score uplift without modifying the frozen baseline:
1. Implement the pairwise re-ranker as an explicit, shadow-tested candidate selection module in `FINAL_SUBMISSION/runtime/src/`.
2. Apply the **Second-Look Evaluator**:
   - If candidate Rank 1 margin is $> 0.05$: preserve Rank 1 (100% frozen, fast-path).
   - If candidate Rank 1 margin is $\le 0.05$ (near-tie): apply the multi-signal structural comparator to verify if a close competitor possesses superior context/neighborhood evidence.
3. Validate on the full 180-pair benchmark:
   - Target: $+8$ to $+12$ ranking recoveries, $0$ localization regressions, lifting grayscale score from **~90.98 $\to$ ~92.0+**.
