# Experiment registry — verified-generator 90+ research loop

Budget: 12 major hypotheses. Checkpoint every 3.
Rules in force: no training/fitting/threshold-selection on `data/phase2_dev`;
no cache; no `pair_id` logic; no network; `FINAL_SUBMISSION*` never modified;
every experiment in its own directory, never overwritten.

Scoring: `phase2/V48_MAX/score_phase2_official.py` (pptx rubric verbatim).
Interpreter: `C:/Python314/python.exe` (sklearn 1.8.0, matches the pickles).

| # | hypothesis | outcome | loc /40 | pose /20 | promoted |
|---|---|---|---|---|---|
| EXP001 | Engine B baseline on verified labels; is retrieval or ranking the bottleneck? | **Ranking.** R4=0/78, R2=36/78 | 21.03 | 9.73 | baseline |
| EXP002 | The V25 learned ranker is anti-correlated with truth on verified labels | **Confirmed.** `context_combined` argmax: +34 recoveries, −2 | **37.74** | **16.51** | gate 3/5 — held |
| EXP003 | Sequential pose search is corrupted by rotation; joint search recovers it | **Confirmed.** scale≤1% 50→69/78, pose credit 0.853→0.962 | — | 19.24 | held |
| EXP004 | pose_v2 + context selector end to end | **53.80/85** vs baseline 36.48 | 24.10 | 11.85 | held |
| EXP005 | Runtime: pose is 95% of cost, not candidate scoring | pose_v3 **2.02×** faster, estimate identical on 28/30 | — | — | in progress |
| EXP006 | Presence gate costs ~13 loc points; refit it on train-corpus data | *awaiting corpus* | — | — | — |

EXP004 is the current best integrated result: loc 24.10 + pose 11.85 +
rejection 7.85 + calibration 10.00 = **53.80/85**, against the Engine B baseline's
36.48/85. Not promoted to any shipping artefact.

## Standing measurements

- Benchmark: `experiments/EXP001_pilot100/data`, 95 pairs (78 present / 17 absent),
  all present labels independently verified (GT NMS rank 1 for 78/78, peak err mean 0.61 px).
- Oracle ceiling on this benchmark: **39.38/40** localization.
- Scale within the rubric's 1% top tier: **50/78**. Exact scale ⇒ pose 19.41/20.
- Runtime median **5.11–5.48 s/pair** — over the 5 s budget. Open item, unowned.

## Negative results (kept, not deleted)

- **STEP 10 Global Alignment Discriminator** — rejected. Replica and GT both ≈0.19
  whole-patch NCC at 3.2× footprint; degradation collapses correlation at every footprint.
- **Fused selectors** (z-sum and product of corr/ctx/grad) — *worse* than
  `context_combined` alone (36.82 vs 37.74). Adding correlation costs 2 pairs.
- **Shortlist-then-rerank**, 44 configurations — no configuration achieves zero
  breakage; the 2 regressions are upstream pose failures, not ranking failures.
- **`dist_to_center`** — 0/78 as a selector. The learned ranker's most-weighted feature
  carries no signal on verified data.
