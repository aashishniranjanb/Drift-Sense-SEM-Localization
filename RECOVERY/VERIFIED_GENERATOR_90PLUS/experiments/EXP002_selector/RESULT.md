# EXP002 — the V25 learned ranker is anti-correlated with truth

**Hypothesis:** R2 dominates (36/78) because `ranker.pkl` was fitted on the
unverifiable `phase2_dev`-like distribution. On verified labels a raw structural
signal should select the correct candidate more often than the learned ranker.

**Result: confirmed, and larger than expected.**
**Status: candidate for promotion, gate not cleanly passed (2 regressions) — see §4.**

## 1. Method

`dump_full_pool.py` runs Engine B once per pair and dumps **all 200 candidates** with
every V25 feature plus sub-pixel refined coordinates (`pool_pilot100.csv`, 19 000 rows).
Selectors are then swept offline on that fixed pool, so every selector sees exactly the
same candidates, pose estimate and refinement. Ground truth is used only to *score*,
never to select. Nothing is fitted — each selector is a plain argmax over one column.

## 2. Selector sweep, full 200-candidate pool, 78 present pairs

| selector | ≤1 px | ≤2 px | ≤5 px | localization /40 |
|---|---|---|---|---|
| **`context_combined`** | **72** | **74** | **74** | **37.74** |
| fuse corr+ctx+grad (z-sum) | 71 | 72 | 72 | 36.82 |
| fuse ctx+grad | 71 | 72 | 72 | 36.82 |
| product corr·ctx | 71 | 72 | 72 | 36.82 |
| `corr_score` (raw NCC) | 71 | 71 | 71 | 36.41 |
| `phase_penalty` (low) | 69 | 71 | 71 | 36.21 |
| `context_128` | 69 | 70 | 70 | 35.79 |
| `grad_ncc` | 69 | 69 | 69 | 35.38 |
| `neigh_cons` | 41 | 41 | 42 | 21.23 |
| **`v25` (baseline learned ranker)** | 40 | 40 | 42 | **21.03** |
| `dist_to_center` (low) | 0 | 0 | 0 | 0.00 |
| *oracle (best candidate in pool)* | *74* | *78* | *78* | *39.38* |

Nine of the twelve hand-written single signals beat the learned ranker. The ranker is
second-from-last, above only `dist_to_center` — the feature it leans on hardest, and
which is worth **zero** here.

Fusion does not help: adding correlation to context *costs* 2 pairs. `context_combined`
alone closes 94% of the gap between the baseline and the oracle ceiling.

## 3. Pose comes along for free

The selector changes which candidate is scored, so pose credit (awarded only where
localization credit > 0) moves with it:

| | localization /40 | pose /20 | sum /60 |
|---|---|---|---|
| `v25` baseline | 21.03 | 9.73 | 30.76 |
| `context_combined` | **37.74** | **16.51** | **54.25** |

## 4. Promotion gate

| gate condition | result |
|---|---|
| ≥5 additional ≤5 px recoveries | **PASS — 34** |
| 0 baseline successes broken | **FAIL — 2** (`v00015`, `v00041`) |
| 0 new absent false positives | n/a — selector does not touch the presence path |
| deterministic | **PASS** — pure argmax, 0 pairs with a tied maximum |
| median runtime ≤5 s | **FAIL — 5.11 s** (pre-existing, unchanged by this experiment) |

The gate is not cleanly passed and I am not relaxing it silently. The two regressions
are recorded in `gate_per_pair.csv` and diagnosed below.

## 5. Why the 2 regressions are not a ranking defect

Shortlist-then-rerank (`shortlist.py`, correlation or v25 proposing, context disposing,
K ∈ {2…200}) produces **zero** zero-breakage configurations — the decoy leads the true
site on every raw signal in both pairs, so no shortlist depth can help.

The cause is upstream. All four remaining failures have clean labels (verifier: GT at
NMS rank 1, peak error < 1 px, margin 0.12–0.20, severity 0). The verifier builds its
template from the **true** (z, θ); Engine B **estimates** them:

| pair | true z | est z | scale err | true θ | est θ | rot err | final err |
|---|---|---|---|---|---|---|---|
| v00015 | 8.7346 | 9.60 | 9.9% | −4.500 | −4.00 | 0.50° | 453 px |
| v00041 | 10.2114 | 11.50 | 12.6% | 4.412 | 4.75 | 0.34° | 313 px |
| v00038 | 9.4494 | 11.00 | 16.4% | 4.649 | 4.75 | 0.10° | 306 px |
| v00053 | 11.9420 | 12.00 | 0.5% | −3.799 | −3.75 | 0.05° | 363 px |

Scale error on the 74 successes: median 0.52%. On the 4 failures: median 11.3%.
**3 of the 4 residual failures are pose-search failures wearing a ranking failure's
clothes.** With a wrong template the correlation surface is wrong, and no selector
reading that surface can recover.

## 6. The next bottleneck, quantified

Engine B's pose grid is coarse: est_scale snaps to 9.6 / 11.0 / 11.5 / 12.0. Scale
lands inside the rubric's 1% top tier for only **50/78** pairs.

> If scale were exact, pose would be **19.41/20** instead of 16.51 — and 3 of the 4
> remaining localization failures would likely resolve as well.

That is EXP003: continuous pose refinement around the grid optimum.

## 7. Limitations

Measured on 95 synthetic verified pairs. `context_combined` compares reference and
search context at several radii, and this generator gives every mat its own lattice
phase and gain — which is what makes context discriminative. Real wafers have distinct
mats too, but the possibility that this generator is *unusually* kind to context is a
live risk and is the first thing the transfer test must probe. Not promoted to any
shipping artefact; `FINAL_SUBMISSION/` and `FINAL_SUBMISSION_GOLDEN/` untouched.

## Artefacts
`pool_pilot100.csv` · `selector_sweep.csv` · `shortlist_sweep.csv` · `gate_per_pair.csv` ·
`sweep.py` · `shortlist.py` · `gate.py` · `../../dump_full_pool.py`
