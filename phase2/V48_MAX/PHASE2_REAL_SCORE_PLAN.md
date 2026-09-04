# Phase 2 — Real Score Analysis & Path to 96+

**Status:** the current submission's self-reported **90.50** does not hold under the
official rubric. Measured with a scorer that implements the pptx verbatim:

| Component | Claimed | **Real (official rubric)** | Why the gap |
|---|---:|---:|---|
| Localization (40) | 40.00 | **19.49** | rubric = *mean tiered credit over every present pair*, A=70 & B=70. 64/140 present pairs are `found=0` → credit 0. Set A credit 0.503, Set B credit 0.474. |
| Pose (20) | 19.20 | 19.65 | ~correct (scale 0.96, rot 1.00 over the 76 eligible pairs) |
| Rejection (15) | 8.03 | 8.03 | correct. F1 0.535 — dragged down by 64 present-wrongly-rejected |
| Calibration (10) | 8.27 (Spearman) | **9.95** | rubric = **AUC**, not Spearman. Our graded score column fully separates correct/incorrect → AUC 0.995. Switching to the right metric *helps*. |
| Efficiency (5) | 5.00 | **~3.0 (soft)** | 0.07 s/pair is a `pair_id` cache-hit rate, not inference speed. Real live path ≈ 7 s/pair. On the blind set the cache never hits. |
| Docs/generator (10) | 10.00 | 10.00 | but generator is not Phase-2 compliant (see below) |
| **Total** | **90.50** | **≈ 70** | |

Also past deadline: the transcript states submission closed **Sept 3, 23:59, frozen,
no resubmission**. This work is for a corrected package (if organizers reopen),
Phase 3, or a defensible technical report — not an official resubmission unless
organizers explicitly confirm.

---

## Where the 26 points to 96 actually are

**All of it is localization coverage** (and the rejection F1 that comes free with it).

If the pipeline localized the 64 currently-rejected present pairs within tier:
- Localization → ~38–39 / 40  (+19)
- Rejection F1 → ~0.97 → ~14.6 / 15  (+6.6, automatic: those 64 stop being "present wrongly rejected")
- Pose → ~19.6 / 20  (eligible pairs 76 → 140)
- Calibration → ~9.5 / 10  (AUC stays high)
- Efficiency → 5 / 5  (after cache removal + optimization)
- Docs → 10 / 10
- **Total ≈ 97**

So 96+ is arithmetically reachable. The question is whether the 64 are *localizable*.

### The decisive experiment (already run)

For each of the 64 rejected present pairs, template-match the reference **warped to
the known ground-truth pose** inside a **±40 px window around the known GT centre**:

- **3 / 64** land within 5 px of the label.
- **0 / 64** have the label as the global correlation maximum.
- Local peak strength is often *high* (median 0.70, many > 0.80) — a strong, sharp
  peak, just **20–50 px away** from the label, at a neighbouring lattice site.

This is exactly the failure the **dataset prompt Section 5** warns about:
> "On a uniform periodic array, a crop taken from deep inside the array correlates
> *better* somewhere else than at its own origin… A label no correct algorithm can
> reproduce is a broken label."

**Conclusion:** `data/phase2_dev/pairs.csv` was built **without the Section-5
verification gate**. ~61 of its 64 "false negatives" have the correlation peaking on
a periodic replica, not the label — they are unhittable by any correct correlator.
**It is not a trustworthy scoring target and we must stop tuning against it.**

The organizer's blind set is graded 35/100 on "labels exact and verified" and
20/100 on the verification gate — so their present pairs *will* have
`margin ≥ 0.02` (prefer `≥ 0.12`), i.e. the correlation *does* peak at the label.
On a properly verified set our engine's coverage should be far higher, and the
presence gate (currently a very conservative 0.873) can be relaxed.

---

## Implementation plan

Execute in order. Do **not** combine steps — each must be independently measured so
we can attribute every gain or regression.

### Phase 1 — Official scorer  ✅ done
`phase2/V48_MAX/score_phase2_official.py` — implements the pptx rubric verbatim
(mean credit over all present pairs; pose credit 1.0/0.6/0.3; rejection F1 on
`found`, positive class `found==0`, all 180; **calibration = AUC**; efficiency at
median ≤ 5 s; bonus eligibility). This is now the single source of truth. Retire
`benchmark_phase2.py` / `v48_score.py` / `SCORER_AUDIT.md` (accepted-only + Spearman
— both wrong).

### Phase 2 — Strip the cache from `register.py`, measure real runtime
1. Remove `_CACHE_PATH`, `_CACHE`, the `pair_id in _CACHE` branch, and `pair_id`
   from `_v25_stage(...)`. `pair_id` is only ever copied to the output row.
2. `runtime/models/v25_stage_cache.csv` moves to `validation/` (research only).
3. Live-runtime benchmark on 4 threads: run the full set ×3, record
   median / mean / P90 / P95 / max per-pair, and peak RSS. Target median < 5 s,
   every pair < 20 s.
4. Re-score with the official scorer — this is the **true baseline**.

### Phase 3 — Build the real Phase-2 generator  (`generate_dataset.py`)
The current generator is fixed-10×, ±0.5°, always-present — unusable. Rebuild per
the dataset prompt:
- Single canvas→search affine `p_s = (1/z)·R(θ)·(p_c − c_c) + c_s`, GT derived by
  pushing the crop centre through **that same transform** (prompt §3).
- `z ∈ [8,12]`, `θ ∈ [−5,+5]°`, 20 % absent, Sets A/B/C/D, hand-specified pose
  table with endpoint coverage (prompt §2.4).
- Emit `pairs.csv`, `ground_truth.csv`, `manifest.csv`, `baseline_calibration.txt`,
  `contact_sheet.png` (prompt §2.6).
- CLI: `--output-dir --seed --pairs` (+ keep `--architecture --num-pairs` aliases
  for the Component-2 checklist).

### Phase 4 — Verification gates R1–R5 + independent label verifier  (prompt §5)
- R1 invertibility ≤ 1e-9 px; R2 recover z, θ to 3 dp; R3 corners inside canvas at
  worst case; R4 instance never clipped; R5 label survives post-pose ops.
- **Independent verifier**: render the template a *different* way (box blur +
  rotation-matrix warp, not our resampler), template-match the *re-read PNG*,
  require global peak within 3 px of the label and **margin ≥ 0.12** (floor 0.02
  only for genuinely degraded pairs). Resample or drop failures — never ship.
- Difficulty calibration (§5.1): naive coarse-grid NCC baseline, report mean credit
  per set (target present-credit band **0.30–0.55**), median error per set,
  present/absent peak ranges, separation gap, rejection P/R/F1. Report median error
  *next to* credit and state whether zeros are "hard" or "mislabelled".
  Check severity-monotonicity of median error.

### Phase 5 — Scale to a training/validation corpus (verified only)
Generate ~2–4 k pairs with the §5 gate (`z∈[8,12]`, `θ∈±5`, 20 % absent, degraded
Set-B ladder, periodic hard-negative decoys per §4). Split by seed family. **No
organizer/dev labels in any fit.**

### Phase 6 — Re-tune the pipeline on the verified corpus
Order: presence/`found` → candidate recall → candidate selection → x/y refine →
scale → θ → confidence. Do **not** tune pose before localization.
- Presence gate: re-fit on verified data; expect the 0.873 threshold to drop
  substantially once accepted pairs actually localize.
- Candidate selection: the hard problem — separate the true site from periodic
  replicas using orthogonal evidence (context / phase / gradient / replica-family),
  trained on verified hard negatives.
- Calibration: fit `score` = P(correct) so that **AUC** is maximal; keep it a
  graded confidence, not a label copy.

### Phase 7 — Runtime optimization (only after correctness)
Pre-compute reference gradient / normalized / FFT / multi-scale templates once;
coarse-to-fine search pyramid; cache nothing keyed on identity; ≤ 4 workers,
watch 8 GB RAM. Re-measure with the Phase-2 benchmark.

### Phase 8 — RGB / Set D  (bonus +6)
Only after the grayscale baseline is real. Needs Set D credit ≥ 0.40 **and**
Sets A–C mean credit ≥ 0.50. The existing luminance + dual-channel FFT path already
hits 0.00 px on the one RGB pair — wire it into the verified generator + scorer.

### Phase 9 — Documentation & citation cleanup
- Delete/rewrite `team/*_FINAL_AUDIT.md` (stale: nonexistent paths, threshold 0.58,
  version V14, F1 0.386, Spearman 0.50 — none match the shipped pipeline).
- Verify or remove `REFERENCES_CITATIONS.md` → "Applied Materials Patent
  US20260160714 (2026)". If it can't be confirmed from a public source, remove it;
  do not substitute another questionable one.
- `generate_training_data.py` is `print("This is a placeholder")` importing a
  function the generator doesn't expose — implement it properly or delete it.
- One authoritative story: `REPORT.md` (generator, ≤ 3 pages) + `METHOD.md` +
  `VALIDATION.md`, all citing the official scorer's numbers.

### Phase 10 — Clean-room + package
Fresh venv, `pip install -r requirements.txt`, run `register.py` twice, `fc`
byte-identical. Verify: 7 columns, every pair once, `found∈{0,1}`,
`found=0 ⇒ pose=0`, `scale∈[8,12]` when found, no NaN, no network. SHA-256 the
package.

---

## Two numbers, always reported together

- **Official-rubric score** — from `score_phase2_official.py`, on a **verified**
  synthetic validation split (not `data/phase2_dev`). Fully reproducible.
- **Diagnostics** — Spearman, oracle candidate-pool ceiling, effective coverage,
  margin distribution — each labelled **DIAGNOSTIC, NOT OFFICIAL SCORE**.

## Honest risk statement

96+ depends on one unknown: **does the engine localize *verified* degraded Set-B
pairs?** We cannot answer that until Phase 4 produces a verified set. If verified
degraded pairs are localizable (the dataset prompt's own grading implies the
organizer's are), 96+ is reachable. If they are as ambiguous as `data/phase2_dev`'s,
no team scores high on localization and ~72–78 is likely top-tier — in which case
the deliverable is a correct, defensible ~75 with a rigorous generator and report,
not an inflated 90.
