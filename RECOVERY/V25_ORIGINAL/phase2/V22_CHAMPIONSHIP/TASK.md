# Phase V22: Championship Fusion — Final Algorithmic Experiment

## MISSION
Maximize the actual Applied Materials Phase 2 competition score.
This is the **FINAL** algorithmic experiment. After V22, no new algorithmic phase will be opened.

---

## FROZEN CONTROL (DO NOT MODIFY)
- `FINAL_SUBMISSION/` + `Drift-Sense-Phase2.zip`
- SHA256: `8637b79ab130a1e6c910d3a5e7c74577c02e3679a688ca80525186d20409ec99`
- **V21 Control is the baseline to beat**

---

## ROOT CAUSE TO FIX
Current system (V21) is over-rejecting: `found=0` on **101/180 cases** when only 40 should be absent.
The V14 gate at `threshold=0.58` is too conservative for the integrated V19+V18-C pipeline,
which produces lower raw NCC/context scores because it explores more of the correlation space.

---

## SCIENTIFIC OBJECTIVE
Build one final **candidate-evidence fusion layer** that jointly decides:
1. `found` (presence gate)
2. `score` (calibrated confidence)

...using the strongest evidence signals already demonstrated, without sacrificing Set B localization or pose.

---

## TRAINING TARGET
`candidate_is_correct` — not merely `present`

### Hard Negative Sources (per-pair, from actual V18/V19 candidate pools)
- Wrong V18-C winner (top-ranked but wrong location)
- High-NCC periodic replica candidates
- Suppressed candidate from V19 Rescue Queue (rescued but ranked wrong)
- Boundary/peripheral candidates
- Random false positives

---

## EVIDENCE FEATURES TO USE
| Feature | Source |
|---|---|
| `corr_score` | V19 NCC peak value |
| `psr` | Phase Sharpness Ratio |
| `peak_margin` | Top-1 vs Top-2 margin |
| `context_32/64/128` | Multi-scale context |
| `phase_residual` | Phase correlation residual |
| `phase_penalty` | Phase consistency penalty |
| `dist_to_center` | Spatial center prior |
| `family_population` | Periodic cluster size |
| `ambiguity_index` | Global periodicity measure |
| `num_peaks_90` | **V20.3 global validator** — Only if validation AUC > 0.85 on unseen split |

### DO NOT USE
- V20-H LogReg classifier
- V20 structural-anchor rule
- Test-set-derived thresholds
- Organizer blind data

---

## EXPERIMENT LADDER

### V22-A: V21 Control Baseline
- Run the frozen V21 pipeline (`production_runner.py` with V19+V18-C+V14)
- Record **exact** competition-style scores on all 180 dev pairs
- This is the benchmark to beat

### V22-B: Feature Extraction + Logistic Fusion
- For each pair in dev set, extract the full evidence vector from the **actual best candidate**
- Train LogReg on `candidate_is_correct` with hard-negatives
- Strict train/val/test split (never re-use test pairs in threshold selection)

### V22-C: Add Global Validator Feature
- Compute `num_peaks_90` for every candidate
- Add to feature set of V22-B
- Compare AUC vs V22-B on val split only

### V22-D: Nonlinear Fusion
- Try `HistGradientBoostingClassifier` (CPU-native, no GPU)
- Do NOT try deep learning

### V22-E: Hard-Negative Mining Iteration
- Re-run training with the actual wrong candidates from V18/V19 pools
- This is the key ablation — does training with rank-aware negatives improve candidate discrimination?

### V22-F: Probability Calibration
- Calibrate final model output via `CalibratedClassifierCV(method='isotonic')`

---

## THRESHOLD SWEEP (CRITICAL)
For `T = 0.05, 0.10, ..., 0.95`:
- Compute `found = 1 if P(correct) >= T else 0`
- Compute full competition score breakdown
- Select `T*` = argmax(total competition score) on VALIDATION split only
- Freeze `T*` before evaluating on test

---

## COMPETITION SCORING FORMULA (implement exactly)

```python
def compute_competition_score(merged_df):
    """
    merged_df: GT + predictions merged on pair_id.
    Returns dict with all score components.
    """
    # Localization (40 pts)
    # loc_err = Euclidean distance from predicted x,y to gt_x,gt_y (PRESENT only, pred_found=1)
    # Credit: <=1px => 1.0, <=2px => 0.8, <=3px => 0.6, <=5px => 0.4, >5px => 0.0
    # Set A weight 0.45, Set B weight 0.55
    # Normalized to 40 pts

    # Pose (20 pts)
    # Scale: <=1% => 1.0, <=2% => 0.75, <=5% => 0.5
    # Rotation: <=0.25deg => 1.0, <=0.5deg => 0.75, <=1deg => 0.5
    # Pose only scored when localization receives credit (loc_err <= 5px and pred_found=1)

    # Rejection (15 pts)
    # F1 across all 180 cases where positive class = found=0 (ABSENT)
    # Note: applies to both predicted and GT found values

    # Calibration (10 pts)
    # AUC of (score) column against per-pair correctness
    # Correctness = 1 if (loc_err <= 5px AND pred_found == gt_found) else 0

    # Efficiency (5 pts)
    # Median runtime <= 5s => full credit, else partial

    # Generator/Docs (10 pts) — fixed at 10 since included
```

---

## ACCEPTANCE GATE (V22 must beat V21 on ALL of these)
| Metric | V21 Baseline | V22 Requirement |
|---|---|---|
| Total Score | to be measured in V22-A | Must exceed |
| Set B ≤5px | to be measured | No degradation |
| PRESENT recall | to be measured | ≥ V21 |
| Rejection F1 | to be measured | ≥ V21 |
| Calibration AUC | to be measured | ≥ V21 |
| Median runtime | < 5s | < 5s |

---

## OUTPUT DELIVERABLES
```
phase2/V22_CHAMPIONSHIP/results/
├── V22_A_control.csv
├── V22_B_logreg.csv
├── V22_C_global.csv
├── V22_D_xgboost.csv
├── V22_E_hardneg.csv
├── V22_F_calibrated.csv
├── V22_ABLATION.csv
├── V22_THRESHOLD_SWEEP.csv
├── V22_FINAL_PREDICTIONS.csv
├── V22_FINAL_SCORE.csv
├── V22_PER_CASE.csv
├── V22_RESULTS.md
└── V22_DECISION.md   ← must end with: KEEP or REJECT
```

---

## FINAL DECISION RULE
- If V22-best > V21 robustly across all gates → **KEEP V22, update FINAL_SUBMISSION**
- If V22 does not beat V21 → **REJECT V22, V21 backup wins**
- **There is no V23.**

---

## ANTI-LEAKAGE RULES (strict)
1. Train/Val/Test splits fixed before any training
2. Transformations of test pairs cannot appear in training
3. `T*` threshold selected on validation only
4. Final evaluation on held-out test with frozen model + `T*`
5. No per-case tuning
