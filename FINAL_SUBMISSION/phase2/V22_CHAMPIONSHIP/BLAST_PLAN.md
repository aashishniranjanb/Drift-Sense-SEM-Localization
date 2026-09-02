# Phase V23 — Championship Blasts (Retrieval 3.0 + Replica Killer)

## BASELINE: V21 → 50.65/100

```
failure_mode              count
PRESENCE_FALSE_NEGATIVE    76   ← gate rejects present case  
PERIODIC_REPLICA           46   ← wrong candidate ranked #1
SUBPIXEL_SUCCESS           18   ← correct
REJECTION_SUCCESS          25   ← correct absence
ABSENCE_FALSE_POSITIVE     15   ← wrong, accepts absence
```

## The Localization Funnel (140 PRESENT cases)

```
140 PRESENT
  └── ~135 GT in raw correlation plane         (oracle ceiling)
       └── ~70/140 GT in Top-50 (V19)          (retrieval ceiling)
            └── ~42/70 correct ranked #1 (V18-C)(ranking ceiling)
                 └── ~18 survive gate           (acceptance)
```

## BLAST 1 — Retrieval 3.0 (R0–R3)

**Goal**: Top-50 GT recall: 67% → 80%+

**Keeper condition**: Top-50 ≥ 78% AND latency ≤ 1.5× V21

### Variants
- **R0**: V19 existing (control, 67.14%)
- **R1**: Increase pool from 200→500, keep dual-queue logic
- **R2**: Add **Periodic Rescue Queue** — explicitly grid-search the correlation plane on detected lattice spacings to find suppressed GT candidates
- **R3**: R2 + adaptive quotas (periodicity-weighted center/periphery split)

---

## BLAST 2 — Replica Killer Ranker (P0–P2)

**Goal**: Conditional Top-1 (when GT in pool): 60% → 75%+

**Keeper condition**: Conditional Top-1 ≥ 70% on validation split

### Key observation from V17 forensics
```
GT candidate:       center_dist≈119px, corr=slightly lower, context=slightly lower
WRONG replica:      center_dist≈245px, corr=slightly higher, context=slightly higher
```

### Relative features (the key innovation)
For each candidate i in pool of N:
- `ncc_i - median(ncc_pool)` — is this candidate unusually high?
- `center_i - median(center_pool)` — is this unusually peripheral?
- `context_i - median(context_pool)`  
- `phase_residual_i - median(phase_pool)`
- `psr_i - median(psr_pool)`
- `ncc_rank`, `phase_rank`, `center_rank` (rank within pool)
- `family_population` (how many similar candidates?)
- `nearest_competitor_ncc_delta` (margin over second-best)

### Model ladder
- **P0**: V18-C control
- **P1**: Logistic ranker with relative + absolute features
- **P2**: HistGradientBoosting with full feature set + hard negatives

### Hard negatives (critical)
For every PRESENT pair, generate training instances:
- ✅ 1× correct GT candidate (label=1)
- ❌ top-1 V18-C winner if wrong (label=0)
- ❌ highest-NCC non-GT candidate (label=0)
- ❌ periodic replica from family clustering (label=0)
- ❌ peripheral boundary candidate (label=0)
- ❌ random non-GT from pool (label=0)

---

## BLAST 3 — Acceptance Gate (2D sweep)

**Only run after BLAST 1+2 are confirmed.**

Replace single threshold with a 2D gate:
```
found = 1 if (presence_score >= T1) AND (ranking_margin >= T2)
```

Sweep T1 × T2 grid on validation, maximize competition score.

---

## Output Format (mandatory per experiment)

Every run must produce this table:

| Metric | V21 | Experiment | Δ |
|---|---|---|---|
| Localization /40 | 4.91 | ? | ? |
| Pose /20 | 19.55 | ? | ? |
| Rejection /15 | 5.83 | ? | ? |
| Calibration /10 | 5.32 | ? | ? |
| Efficiency /5 | 5.00 | ? | ? |
| **TOTAL /100** | **50.65** | ? | ? |
| PRESENT recall | 45.7% | ? | ? |
| Top-50 GT recall | 67.14% | ? | ? |
| Cond. Top-1 | 60.0% | ? | ? |
| Set A ≤5px | — | ? | ? |
| Set B ≤5px | — | ? | ? |

---

## Kill Conditions
- Blast 1 R1–R3: If Top-50 < 70% → REJECT immediately
- Blast 2 P1–P2: If cond. Top-1 ≤ 60% → REJECT immediately
- Integration: If total score ≤ 50.65 → REJECT, V21 wins
