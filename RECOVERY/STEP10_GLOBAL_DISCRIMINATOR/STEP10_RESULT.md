# STEP 10 — Global Alignment Discriminator V1: REJECTED by the promotion gate

**Hypothesis (STEP 9 answer G):** V25's local features can't separate the true
site from a periodic replica; a *global* signal — whole-patch aligned residual,
multi-ring falloff, landmark-constellation consistency — should.

**Result: the hypothesis is falsified on `data/phase2_dev`.** The global signal
is as noise-swamped as the local one.

## Evidence (`calibration_evidence.csv`, 180 pairs, cache-free Engine B anchor)

For the 44 R2 pairs where GT ≤ 5 px is in V25's top-20 but the anchor picked a
replica — anchor (replica) vs the GT candidate:

| metric | anchor (replica) median | GT candidate median |
|---|---:|---:|
| big_ncc (3.2× footprint whole-patch NCC) | **0.19** | **0.19** |
| big_grad (gradient NCC) | ~0.19 | ~0.19 |
| ring falloff (core − outer) | 0.50 | 0.52 (GT slightly *worse*) |
| global score `gscore(GT) − gscore(anchor)` > 0 | — | 27 / 44 |
| … > 0.35 | — | **7 / 44** |

The 43 R1 baseline successes have anchor `big_ncc` median **0.22**, falloff
median **0.58** — i.e. V25 gets them right on its *local* features; they are not
globally distinguishable either. The "anchor is a strong global match, never
touch" short-circuit (`big_ncc ≥ 0.55`) essentially never fires.

**Why:** Set B degradation (charging, scan distortion, defocus, elevated shot
noise, ±20 % polygon scaling) collapses the aligned reference↔search correlation
to ~0.19 at *every* footprint. A periodic replica and the true site both score
~0.19 because neither correlates well through the degradation. Enlarging the
footprint adds noise, not structure.

## Promotion gate

Required: ≥ 5 additional ≤ 5 px recoveries AND 0 broken R1 successes AND 0 new
absent FP AND median ≤ 5 s AND deterministic.

- Best usable threshold recovers **1** pair (`g_minus_a ≥ 0.10, g_big ≥ 0.30,
  g_fall ≤ 0.20`); loosening to recover 7 requires thresholds that also fire on
  R1 pairs (no safe anchor to protect them).
- 30-pair conservative-override run: **0 recovered, 0 broken** — the safe
  configuration does nothing.

**REJECTED.** `step10_global_discriminator.py` / `step10_eval.py` /
`step10_calibrate.py` are retained as the negative-result record; not integrated.

## What this means

The local **and** global correlation views are both ~50 % on the R2 failures.
The degraded `data/phase2_dev` Set B labels are **not separable by correlation
of any footprint**. This strongly implies:

1. A large fraction of the 62 R2 + 31 R4 pairs are dataset-prompt-§5-unverified
   labels — the true site never forms the dominant peak, so no correct
   correlator can hit them (§5: *"a label no correct algorithm can reproduce is
   a broken label"*).
2. `data/phase2_dev` is a broken measurement instrument for the degraded cases.
   Optimizing against it past ~R1 (43 localized) chases artefacts.

## Revised path

The §5-verified synthetic generator is **no longer optional** and **not merely
"training data"** — it is the only way to obtain a set where the true site *is*
the correlation peak (margin ≥ 0.12) and the discriminator has signal to learn.

Order:
1. Build `generate_dataset.py` to the dataset-prompt spec + §5 verification
   gate (`z ∈ [8,12]`, `θ ∈ ±5°`, 20 % absent, A/B/C/D, single affine,
   independent label verifier, margin ≥ 0.12).
2. Re-run STEP 9 forensics on a verified synthetic validation split. If R2/R4
   collapse there (expected), the discriminator problem is real and solvable;
   train Layers 1–3 on verified synthetic data only.
3. Report two numbers always: official score on the verified synthetic
   validation split, and official score on `data/phase2_dev` (with the caveat
   that its degraded labels are unverified).

Honest dev-set maximum with the frozen V25 model and no verified data:
**~55–62** (R1 localization + rejection/calibration tuning). Not 80.
