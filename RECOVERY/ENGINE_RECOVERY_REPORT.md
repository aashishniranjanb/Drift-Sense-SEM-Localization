# Engine Recovery Report

**Branch:** `engine-recovery` · **Scorer:** `phase2/V48_MAX/score_phase2_official.py`
(official pptx rubric: localization = mean tiered credit over **all 70 Set-A + all
70 Set-B** present pairs, `40·(0.45·A + 0.55·B)`; pose credit 1.0/0.6/0.3 where
localization credit > 0; rejection F1 on `found` across all 180, positive class
`found==0`; **calibration = ROC AUC(score, correctness)·10**; efficiency 5 at
median ≤ 5 s).

**Constraints held for every engine:** run live from image pixels only, no cache,
no `pair_id` used to select an algorithm or result, no organizer labels used, no
network. `data/phase2_dev/pairs.csv` used strictly for scoring.

---

## The question

`FINAL_SUBMISSION/verification/predictions.csv` localizes **76/140** present pairs
(→ ~70/100 official). It was produced **with** the `pair_id` cache. Was that
performance the original V25 engine, or an artifact of the cached
`v28_final_predictions.csv` chain?

## The answer

**It was the cache.** No live engine reproduces it.

| Engine | A loc | B loc | Loc /40 | Pose /20 | Rej F1 | Rej /15 | Cal AUC | Cal /10 | Eff median | **Total** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A** current pipeline, cache removed | 14/70 | 5/70 | 4.67 | 19.58 | 0.391 | 5.87 | 0.800 | 8.00 | 4.44 s | **53.1** |
| **B** original V25, recovered, live | 19/70 | 9/70 | 7.06 | 19.57 | 0.389 | 5.84 | 0.589 | 5.89 | 4.35 s | **53.4** |
| **C** V25 + V39 pose + V41 confidence | 19/70 | 9/70 | 7.06 | 19.57 | 0.389 | 5.84 | 0.589 | 5.89 | 4.36 s | **53.4** |
| REF cached `verification/predictions.csv` | 40/70 | 36/70 | 19.49 | 19.65 | 0.535 | 8.03 | 0.995 | 9.95 | — | 70.1 |

- `RECOVERY/V25_ORIGINAL/` `ranker.pkl` / `presence.pkl` are **byte-identical**
  (md5 `f63bea4a…` / `f853b400…`) to the ones in `FINAL_SUBMISSION`. The models
  never drifted.
- The recovered original V25 (Engine B) localizes **28/140** live — better than
  the current pipeline's 19/140, because the current pipeline stacks a drifted
  V28-C `0.873` gate + V47 validator + V48 calibration on top of V25 and those
  layers *cost* live localization. But 28/140 is still far from the cached
  76/140.
- The cached 76/140 came from `v28_final_predictions.csv`, a V28-era
  re-ranking/re-gating layer that is **not present in the V25 backup** and is not
  reproducible from pixels with any recovered code.

## Reconciliation with earlier numbers

Every figure ever reported ≥ ~55 for this system is an artifact of one of:
- **accepted-only localization scoring** (dividing by the 76 accepted pairs
  instead of all 140 present) — inflates ~4.7/40 to "40/40";
- **the `pair_id` cache** — replays `v28_final` for the dev pair_ids;
- **Spearman instead of AUC** for calibration;
- **analytic projection** (`score_comparison.md`, "V54 91.040", "95–98").

The honest, reproducible-from-pixels score is **~53**.

## Component read

- **Localization (4.7–7.1 / 40):** the binding constraint. Only 19–28 of 140
  present pairs land within any credit tier live.
- **Pose (~19.6 / 20):** strong wherever localization gets credit — scale credit
  ~0.96–0.99, rotation ~0.97–1.0. Not the problem.
- **Rejection (5.8–5.9 / 15):** F1 ~0.39. Dragged down by 80–104 present pairs
  wrongly rejected. Improves automatically with localization.
- **Calibration:** raw V25 presence score → AUC 0.59 (Engine B). The V48 graded
  score column → AUC **0.80** (Engine A). This is the one layer that legitimately
  helps.
- **Efficiency (5 / 5):** 4.35–4.44 s/pair median, max < 5.6 s. No timeouts.

## Recovery anchor decision

STEP 8 gate — "if original V25 restores *substantially* better live
localization, freeze its extraction/ranking as the anchor." Engine B is
**+9 pairs** over Engine A, not substantial. But it is the cleaner base (pure V25,
no drifted V28-C/V47/V48 localization layers), so:

**Anchor = Engine B's localization path (original V25 candidate extraction +
ranker + `0.843` presence gate), with Engine A's V48 graded `score` column for
calibration.** Hybrid projected: loc 7.06 + pose 19.57 + rej 5.84 + cal 8.00 +
eff 5 + docs 10 ≈ **55.5**.

## Next (STEP 9–12)

1. Dump top-20 candidates per present pair from the Engine-B anchor with rank,
   x/y, NCC, gradient, context, phase, neighbourhood, scale, theta, distance to
   GT, margin.
2. Separate the ranking failures (GT candidate present in top-20 but not
   selected) from the retrieval failures (GT not in top-20 at all).
3. Implement **one** structural discriminator — full candidate-aligned
   reference/search comparison + multi-ring residual + geometric/landmark
   consistency. No ML yet.
4. Promotion gate: ≥ 5 additional ≤ 5 px recoveries, 0 broken baseline
   successes, 0 new absent false positives, median runtime ≤ 5 s. Fail → delete.

Do not touch rejection or calibration until localization improves. Report
measured numbers only.
