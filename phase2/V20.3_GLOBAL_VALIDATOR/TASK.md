# Phase V20.3: Global Frequency/Phase Validator (Shanganidhi Stream)

## Scientific Question
> **A genuine match should produce a coherent correspondence in the global phase/correlation structure, whereas an isolated periodic replica may produce locally convincing but globally ambiguous evidence. Does global spectral and correlation-plane analysis separate PRESENT from ABSENT?**

## Experiment Scope
Rather than inspecting the local candidate patch or scalar scores, we extract evidence from the full cross-correlation and phase spaces.

### Useful Evidence to Evaluate:
1. **Correlation-plane entropy** (is the plane noisy or highly structured?)
2. **Peak concentration** (is energy focused in one location or spread across a lattice?)
3. **Peak-to-secondary-peak structure** (margin and spatial distribution)
4. **Phase coherence**
5. **Phase residual distribution**
6. **Number of equivalent peaks**
7. **Peak lattice regularity** (do the secondary peaks form a perfect grid, indicating periodicity?)
8. **Spectral energy distribution**

## Execution Ladder
- **V20.3-A (Global Extraction)**: Compute the global features across the entire `phase2_dev/pairs.csv` dataset. Maintain strict separation of Set A (Nominal), Set B (Degraded), and Set C (Absent).
- **V20.3-B (Feature Ablation)**: Test individual features and subsets for separability using the same rules as V20-G.
- **V20.3-C (Calibrated Classifier)**: If a generalizable feature set is found, train a strict LogReg classifier (fixed train/val/test splits, no per-case tuning).

## Acceptance Gate
- Primary Target: $F_1 \ge 0.90$
- PRESENT Recall $\ge 0.95$ 
- Set C FPR substantially below 0.77 (must reject hard negatives)
- No unacceptable Set A/B degradation

## Output Deliverables
Record results in `phase2/V20.3_GLOBAL_VALIDATOR/results/`.
- `V20_3_ABLATION.csv`
- `V20_3_LOGREG.csv`
- `V20_3_RESULTS.md`
- `V20_3_DECISION.md` (KEEP / MODIFY / REJECT)
