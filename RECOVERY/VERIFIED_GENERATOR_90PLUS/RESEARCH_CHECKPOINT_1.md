# Research checkpoint 1 — after EXP001–EXP003

Scope: the first three hypotheses of the verified-generator research loop.
Everything below is measured on 95 independently verified synthetic pairs
(78 present / 17 absent). Nothing here is an official competition score, and
nothing was fitted on `data/phase2_dev`.

## The one-paragraph version

The V25 pipeline's problems were misdiagnosed for the whole history of this
project, because the instrument used to diagnose them — `data/phase2_dev` — has
labels that no correct algorithm can reproduce. Measured against labels that are
independently verifiable, retrieval is perfect (the correct site is in the
candidate pool for **78/78** present pairs), and the two components everyone
assumed were solved are the ones that are broken: the learned ranker actively
demotes the true site, and the sequential pose search collapses under the
dataset's own rotation range. Fixing both, with no new model and no training,
moves the rubric total from **36.48** to **53.80** of 85.

## What each experiment established

**EXP001 — the bottleneck is ranking, not retrieval.**
R4 (GT never retrieved) = **0/78**, against 31/140 on `phase2_dev`. R2 (GT
retrieved, wrong candidate chosen) = **36/78**. Every historical effort aimed at
"finding more candidates" was aimed at a non-problem created by broken labels.

**EXP002 — the learned ranker is anti-correlated with truth.**
Sweeping every candidate signal over the full 200-candidate pool: the learned
ranker scores **21.03/40**, raw `context_combined` scores **37.74/40**, and the
oracle ceiling is 39.38/40. Nine of twelve hand-written signals beat the ranker.
It places second-from-last, above only `dist_to_center` — the feature it weights
most heavily and which is worth **0/78** here. `ranker.pkl` learned the artefacts
of the broken training distribution.

**EXP003 — the pose search is corrupted by rotation.**
The coarse scale sweep runs at θ=0, so under the dataset's ±5° rotation the stage
that picks the scale branch is unreliable. Median scale error rises **0.28% →
2.22%** from |θ|≤1° to |θ|∈(3,4°], where 60% of pairs miss the 2% tier. Joint
search (`pose_v2`) takes scale-within-1% from **50/78 → 69/78** and pose credit
from **0.853 → 0.962**, and removes the |θ| dependence entirely (pairs missing the
2% tier at |θ|>2°: 18 → 2).

## Score movement

| | loc /40 | pose /20 | rejection /15 | calibration /10 | total /85 |
|---|---|---|---|---|---|
| Engine B baseline | 15.18 | 7.31 | 4.78 | 9.21 | **36.48** |
| EXP004 (pose_v2 + context selector) | 24.10 | 11.85 | 7.85 | 10.00 | **53.80** |

## Negative results, kept

- **Global Alignment Discriminator** (STEP 10) — rejected; replica and GT both
  ≈0.19 whole-patch NCC at 3.2× footprint.
- **Fused selectors** — z-sums and products of corr/ctx/grad are *worse* than
  `context_combined` alone (36.82 vs 37.74).
- **Shortlist-then-rerank**, 44 configurations — none achieves zero breakage.
- **Effective-scale reporting** (`ref_w/tw` instead of the grid value) — mean
  scale credit 0.664 → 0.662, i.e. nothing.
- **K=8 shortlist for runtime** — correct (identical accuracy to K=200) but
  worthless: profiling showed context was never the cost. Pose is 95% of runtime.

## Open problems, stated plainly

1. **The presence gate is now the binding constraint.** It rejects 31 of 78
   present pairs, holding localization to 24.10/40 when the selector alone
   achieves 37.74/40 — roughly **13 points left on the floor**. Calibration AUC is
   **1.000**, so the ordering is already perfect and only the cut point is wrong.
   `presence.pkl` was fitted in V25-ranker feature space and its 0.843 threshold
   does not transfer. Fixing this requires held-out generator data with disjoint
   seeds; a 400-pair training corpus is generating now. It will **not** be fixed by
   tuning a threshold on the evaluation set.
2. **Runtime.** 5.65 s/pair median against a 5 s budget. `pose_v3` (rotate a
   pre-decimated reference; run coarse branch selection downsampled, refine at
   full resolution) currently measures **2.0× faster** with one branch-selection
   regression under investigation.
3. **Transfer is unproven.** Every number here is from a synthetic generator.
   `context_combined` may be flattered by the per-mat phase and gain variation
   this generator introduces. The transfer test against `data/phase2_dev` as an
   untouched external diagnostic has not been run.

## Budget

5 of 12 hypotheses used (EXP001–EXP005). Next: presence/calibration fitted on the
training corpus (EXP006), then the transfer test.
