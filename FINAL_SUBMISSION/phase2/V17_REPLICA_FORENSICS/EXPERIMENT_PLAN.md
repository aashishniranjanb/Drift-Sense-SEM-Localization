# Phase V17: Experiment Plan

## Protocol
1. **Target Population**: The 35 `PERIODIC_REPLICA` cases identified in `results/CONTROL_V16/failure_taxonomy.csv`.
2. **Extraction Protocol**:
   - Run V16 pose search and candidate extraction ($K=200 \to \text{Rescue Queue} \to \text{Top-50}$).
   - Extract candidate records for:
     - $C_{GT}$: Ground Truth candidate (within $\le 5\text{ px}$).
     - $C_{1}$: Winning candidate (#1).
     - $C_{2}$: 2nd ranked candidate (#2).
     - $C_{3}$: 3rd ranked candidate (#3).
3. **Features Computed per Candidate**:
   - `corr_score`: Raw template cross-correlation peak.
   - `psr`: Peak-to-sidelobe ratio.
   - `phase_residual`: Phase correlation residual error.
   - `phase_penalty`: Phase consistency variance.
   - `context_32`, `context_64`, `context_128`, `context_combined`: Dynamic multi-scale structural context NCC.
   - `dist_to_center`: Euclidean distance from candidate center to search FOV center $(sw/2, sh/2)$.
   - `nearest_edge_dist`: Distance to nearest Canny edge.
   - `nearest_cut_dist`: Distance to nearest gate cut / defect boundary.
   - `row_spacing`, `col_spacing`: Intra-family lattice pitch distances.
   - `local_density`: Concentric mean intensity.
   - `family_population`: Size of periodic cluster.
4. **Analysis Protocol**:
   - Compute paired feature deltas: $\Delta(f) = f(C_{GT}) - f(C_{1})$.
   - Perform paired t-tests and compute empirical win-rates.
   - Categorize every failure case into exact root-cause failure mechanisms.
