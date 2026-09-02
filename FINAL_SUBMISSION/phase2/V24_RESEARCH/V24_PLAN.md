# V24 Research Plan: Extracting True Geometric Disambiguation

## Context
Blast 1/2 showed that extending the retrieval pool to 200 pushes Ground Truth (GT) recall from 74% to 88% (123/140 cases). However, V18-C ranking performance simultaneously collapses from 47% to 35%. 

Our initial test (V24-A Pool-Invariant Ranker) replacing `family_population` with `family_ratio` improved this drop (recovering up to 40.6%), but it didn't solve the core issue: at K=200, the pool is flooded with deep periodic false-replicas that have identical local signatures.

## New Direction
As you highlighted, we need new physical evidence, not just re-weighting scalar features. We will branch away from the V21 baseline to experiment safely.

### 1. V24-C: Scale/Rotation Response Surface (Perturbation Consistency)
**Hypothesis:** A true GT match creates a coherent, sharp optimum in the NCC response surface (scale vs rotation). A false replica driven by aliased periodicity will have a malformed response surface (e.g. sharp in x/y but structurally unstable to small scale perturbations).

*   **Extraction:** For top candidates, we perturb the estimated scale by ±2% and rotation by ±1°.
*   **Features:** `scale_stability` (NCC drop-off), `rotation_stability`, `peak_curvature`.

### 2. V24-D: Global Lattice Consistency
**Hypothesis:** Replicas occur in a rigid 2D lattice spanning the die. The correct instance is the one most consistent with this global lattice *and* the expected die boundaries/architecture.

*   **Extraction:** We will fit a 2D periodic lattice model to the candidate pool (estimating `pitch_x`, `pitch_y`, `origin_x`, `origin_y`).
*   **Feature:** `lattice_residual` — the spatial error of a candidate against the closest perfect lattice node. We expect GT to sit perfectly on the lattice, whereas some high-NCC false matches are non-periodic noise clusters.
*   **Extension:** The relative position of the candidate within the detected lattice vs expected image center.

### Next Steps
I have created the `phase2/V24_RESEARCH/` branch. I will script V24-C (Response Surface) to see if local perturbation stability offers separation between GT and the wrong replicas that trick V18-C.
